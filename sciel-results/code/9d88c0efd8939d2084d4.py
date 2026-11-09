"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _initial_hex_lattice(n):
    """Generate a hexagonal lattice of points inside the unit disk, return first n."""
    # We want to pack n points. Use a hexagonal grid with spacing s.
    # Area of hex cell ~ s^2 * sqrt(3)/2. Disk area = pi.
    # n * s^2 * sqrt(3)/2 <= pi  =>  s <= sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.95
    
    # Generate grid points
    # For hex packing, rows are offset
    # Let's generate a rectangular grid of hex points and clip to disk
    max_r = 1.0
    # Number of points along diameter ~ 2/s
    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            r = np.hypot(x, y)
            if r <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: spiral
        theta = np.linspace(0, 2 * np.pi * 10, n)
        r = np.linspace(0, 1, n)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        # Sort by radius to get a nice set, take first n
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')
        points = points[idx[:n]]
    return points


def _compute_score(points):
    """Compute the packing score: min pair distance / 2, min distance to boundary."""
    n = len(points)
    # Pairwise distances using KDTree for efficiency
    tree = cKDTree(points)
    # Query for nearest neighbor for each point
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1])
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair / 2.0, min_boundary)


def _softmin_obj_and_grad(x, n, beta=100.0):
    """
    Smooth surrogate objective using log-sum-exp for min pair distance and min boundary.
    Returns (objective_value, gradient).
    We maximize score = min(min_pair/2, min_boundary).
    Minimize -score.
    Using soft-min: soft_min(a, b, beta) ≈ -1/beta * log(exp(-beta*a) + exp(-beta*b))
    For multiple values, soft_min over all values.
    
    Actually, score = min( min_pair/2, min_boundary )
    Let A = min_pair/2, B = min_boundary.
    We want to maximize min(A, B).
    Equivalent to minimizing -min(A, B).
    
    Soft-min of A and B: -1/beta * log(exp(-beta*A) + exp(-beta*B))
    This is differentiable.
    
    But A itself is a min over pairs. So we need soft-min over all pair distances/2 and all boundary distances.
    Let values = [d_ij/2 for all i<j, 1-|p_i| for all i]
    soft_min(values) = -1/beta * log(sum(exp(-beta * v)))
    We want to maximize this, so minimize -soft_min(values).
    """
    points = x.reshape(n, 2)
    radii = np.linalg.norm(points, axis=1)
    
    # Boundary distances
    boundary_dists = 1.0 - radii
    
    # Pair distances
    # For efficiency, use KDTree to find nearest neighbors, but for soft-min we need all pairs?
    # No, soft-min over all pairs is expensive. 
    # Alternative: use soft-min only over the "active" set? 
    # Or use a different approach: minimize max of violations?
    # Let's stick to the direct soft-min over all constraints for correctness, but it might be slow.
    # For n=300, number of pairs is ~45000. exp(-beta * v) might underflow.
    
    # Let's try a different strategy:
    # Minimize: max( max_i (|p_i| - 1), max_{i<j} (0 - |p_i - p_j|/2 + r) ) for some r?
    # No, we want to maximize r such that |p_i - p_j| >= 2r and |p_i| <= 1-r.
    # This is a feasibility problem for r.
    # We can use binary search on r and check feasibility?
    # Or use an epigraph formulation.
    
    # Let's try the soft-min approach but carefully.
    # To avoid underflow, we can shift by the min value.
    # Let m = min(values). Then soft_min = m - 1/beta * log(sum(exp(-beta*(v - m))))
    
    # Compute all values
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    pair_vals = dist / 2.0
    
    # Flatten pair values (only upper triangle to avoid duplicates)
    # But for soft-min, duplicates don't change the result much if beta is large, 
    # but it changes the gradient weighting. 
    # Actually, min is min. Duplicates don't matter for the value.
    # For gradient, the weight of each term in the sum matters.
    # If we include both (i,j) and (j,i), the weight is doubled for that pair.
    # This is equivalent to a different beta. 
    # Let's use upper triangle to be clean.
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals_flat = pair_vals[i_idx, j_idx]
    
    # Boundary values
    boundary_vals = boundary_dists
    
    # Combine
    all_vals = np.concatenate([pair_vals_flat, boundary_vals])
    
    # Soft-min computation
    m = np.min(all_vals)
    # Shift to prevent underflow
    shifted = all_vals - m
    # If beta is large, exp(-beta * shifted) can underflow for large shifted.
    # This is fine, it just means those terms are negligible.
    # But if all shifted are 0, exp(0)=1.
    
    # Compute log-sum-exp: log(sum(exp(-beta * shifted)))
    # = -beta * shifted_max + log(sum(exp(-beta * (shifted - shifted_max))))
    # Here shifted_max = 0 (since m is min, shifted >= 0, max is 0 if any v==m)
    # Wait, shifted = v - m >= 0. So -beta * shifted <= 0.
    # The max of -beta * shifted is 0 (when v=m).
    # So log_sum_exp = log(sum(exp(-beta * shifted)))
    
    # To be safe against underflow in sum, we can just compute it.
    # If beta=100, and shifted=0.1, exp(-10) ~ 4.5e-5.
    # If shifted=1.0, exp(-100) ~ 3.7e-44.
    # Sum will be dominated by terms with small shifted.
    
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # All underflowed? Unlikely if m is included.
        sum_exp = 1e-300
    
    log_sum_exp = np.log(sum_exp)
    
    soft_min_val = m - log_sum_exp / beta
    
    # Objective is -soft_min_val
    obj = -soft_min_val
    
    # Gradient
    # d(soft_min)/dv_k = -1/beta * d(log(sum(exp(-beta*v)))/dv_k)
    # = -1/beta * (1/sum) * sum_j (-beta * exp(-beta*v_j) * delta_jk)
    # = -1/beta * (1/sum) * (-beta * exp(-beta*v_k))
    # = exp(-beta*v_k) / sum(exp(-beta*v_j))
    
    # So d(soft_min)/dv_k = w_k, where w_k = exp(-beta*v_k) / sum(exp(-beta*v_j))
    # And d(obj)/dv_k = -w_k
    
    weights = exp_terms / sum_exp
    
    # Now map weights back to points
    grad_points = np.zeros_like(points)
    
    # Boundary part
    # v_k for boundary is 1 - |p_k|
    # d(1 - |p_k|)/dp_k = -p_k / |p_k|
    # So d(obj)/dp_k = -w_boundary_k * (-p_k / |p_k|) = w_boundary_k * p_k / |p_k|
    
    # Pair part
    # v_ij = |p_i - p_j| / 2
    # d(|p_i - p_j|/2)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    # d(obj)/dp_i = -w_ij * (p_i - p_j) / (2 * |p_i - p_j|)
    
    # Loop over pairs? Too slow for 45k pairs in Python.
    # We need vectorized gradient accumulation.
    
    # Gradient from boundary
    b_weights = weights[len(pair_vals_flat):]
    # b_weights corresponds to boundary_vals
    # grad_boundary = sum_k w_k * (p_k / |p_k|)
    # But we need to accumulate into grad_points
    # grad_points[k] += b_weights[k] * p_k / |p_k|
    
    safe_radii = np.where(radii > 1e-12, radii, 1.0)
    grad_boundary = (b_weights / safe_radii)[:, np.newaxis] * points
    grad_points += grad_boundary
    
    # Gradient from pairs
    # For each pair (i, j) with weight w_ij:
    # grad_points[i] -= w_ij * (p_i - p_j) / (2 * |p_i - p_j|)
    # grad_points[j] += w_ij * (p_i - p_j) / (2 * |p_i - p_j|)
    
    # This is hard to vectorize easily without loops or sparse matrices.
    # Given the time constraints, maybe the soft-min approach is too complex to implement efficiently in pure NumPy for the gradient.
    
    # Let's reconsider the "active set" approach from the original code, but smoothed.
    # Or just use the original gradient but with a softer min?
    
    # Actually, the original code's gradient is 0 almost everywhere, which is bad for L-BFGS.
    # L-BFGS needs a non-zero gradient to make progress.
    
    # Let's try a simpler smooth objective:
    # Minimize: max_pair_violation + max_boundary_violation + penalty * (sum of squares of violations)?
    # No, that doesn't maximize the radius.
    
    # Let's go back to the idea of using a differentiable approximation of the min.
    # The issue is the gradient calculation.
    # Can we use finite differences? No, too slow.
    
    # Let's try to implement the pair gradient vectorized.
    # We have i_idx, j_idx, and weights w.
    # diff = points[i_idx] - points[j_idx]
    # dist = np.linalg.norm(diff, axis=1)
    # grad_vec = (diff / (2 * dist))[:, np.newaxis] * w[:, np.newaxis]
    # Then we need to scatter grad_vec to grad_points.
    # grad_points[i_idx] -= grad_vec
    # grad_points[j_idx] += grad_vec
    
    # This can be done with np.add.at
    
    p_weights = weights[:len(pair_vals_flat)]
    diff_vec = points[i_idx] - points[j_idx]
    dist_vec = np.linalg.norm(diff_vec, axis=1)
    safe_dist = np.where(dist_vec > 1e-12, dist_vec, 1.0)
    
    # Gradient contribution for each pair
    # d(obj)/dp_i = -w * (p_i - p_j) / (2 * dist)
    # d(obj)/dp_j = +w * (p_i - p_j) / (2 * dist)
    
    grad_pair_vec = (diff_vec / (2.0 * safe_dist))[:, np.newaxis] * p_weights[:, np.newaxis]
    
    # Scatter
    np.add.at(grad_points, i_idx, -grad_pair_vec)
    np.add.at(grad_points, j_idx, grad_pair_vec)
    
    return obj, grad_points.flatten()


def _objective_soft(x, n, beta=100.0):
    """Objective function for optimization: negative soft-min score."""
    obj, _ = _softmin_obj_and_grad(x, n, beta)
    return obj


def _objective_soft_grad(x, n, beta=100.0):
    """Gradient of the objective."""
    _, grad = _softmin_obj_and_grad(x, n, beta)
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 15s safety margin
    
    # Step 1: Generate initial hexagonal lattice
    points = _initial_hex_lattice(N)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Step 2: Use L-BFGS-B to optimize with soft-min objective
    x0 = points.flatten()
    
    # Try different beta values
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 20.0:
            break
            
        try:
            result = minimize(
                _objective_soft,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                jac=_objective_soft_grad,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                x0 = new_points.flatten()
        except Exception:
            pass
    
    # Step 3: If time remains, try a few random restarts with local optimization
    while time.time() < deadline:
        # Random perturbation of best points
        noise_scale = 0.01
        x0 = best_points.flatten() + np.random.normal(0, noise_scale, size=N * 2)
        
        try:
            # Use a moderate beta for speed
            beta = 100.0
            result = minimize(
                _objective_soft,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                jac=_objective_soft_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    # Final validation: ensure all points are inside and non-overlapping
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0):
        # Clip points that are slightly outside
        mask = radii > 1.0
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
