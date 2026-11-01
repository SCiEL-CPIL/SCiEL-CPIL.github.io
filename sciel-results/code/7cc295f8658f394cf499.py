import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the actual score: min of half min pairwise distance and min margin to boundary."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dists_to_center = np.sqrt(np.sum(points**2, axis=1))
    if np.any(dists_to_center > 1.0 + 1e-12):
        return 0.0
        
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_margin = 1.0 - np.max(dists_to_center)
    r = min(min_pair_dist / 2.0, min_margin)
    if r < 0:
        return 0.0
    return r

def _objective_and_gradient(points_flat, beta):
    """
    Objective: Minimize sum of soft-min terms.
    We want to maximize min(d_ij, 1-r_i).
    Equivalent to minimizing -min(...).
    Soft approximation of min(x) is -1/beta * log(sum(exp(-beta*x))).
    So we minimize: 1/beta * log(sum(exp(-beta*d_ij) + sum(exp(-beta*(1-r_i)))))
    """
    points = points_flat.reshape(N, 2)
    
    # Calculate pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10) # Avoid self-interaction
    
    # Calculate distances to boundary
    radii = np.sqrt(np.sum(points**2, axis=1))
    margins = 1.0 - radii
    
    # Values for soft-min
    # We combine pairwise distances and boundary margins
    # Note: The score is min(min_pair_dist/2, min_margin).
    # To be consistent, we should treat the "value" to maximize as min(min_pair_dist/2, min_margin).
    # Let's define values v_k.
    # For pairs: v = d_ij / 2.0
    # For boundary: v = 1 - r_i
    
    pair_dists = dists / 2.0
    # Only take upper triangle to avoid double counting
    i_upper = np.triu_indices(N, k=1)
    pair_vals = pair_dists[i_upper]
    
    boundary_vals = margins
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Compute log-sum-exp for stability
    # log(sum(exp(-beta * v)))
    # max_v = np.max(all_vals)
    # log_sum = np.log(np.sum(np.exp(-beta * (all_vals - max_v)))) + (-beta * max_v)
    # obj = log_sum / beta
    
    # Direct computation with clipping to avoid overflow/underflow issues in exp
    # Since we want to maximize min, and v are positive, -beta*v are negative.
    # If v is large, exp is small.
    
    exp_terms = np.exp(-beta * all_vals)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        sum_exp = 1e-300
        
    obj = np.log(sum_exp) / beta
    
    # Gradient
    # d/dx log(sum(exp(-beta*v))) = (1/sum) * sum(exp(-beta*v) * (-beta * dv/dx))
    # d/dx (log(sum)/beta) = (1/beta) * (1/sum) * sum(exp(-beta*v) * (-beta * dv/dx))
    # = -(1/sum) * sum(exp(-beta*v) * dv/dx)
    
    weights = exp_terms / sum_exp
    
    # Gradient of pair distances w.r.t points
    # d/dp_i (d_ij / 2) = (p_i - p_j) / (2 * d_ij)
    # We need to sum over all pairs.
    grad = np.zeros((N, 2))
    
    # Pairwise gradient
    # dists is NxN. diff is NxNx2.
    # We need to accumulate contributions from all j != i.
    # Let's iterate or use matrix ops.
    # For efficiency, we can use the fact that gradient of distance is unit vector.
    
    # Create mask for valid pairs
    # We computed weights for upper triangle only in pair_vals.
    # Let's recompute weights for all pairs to make gradient calculation easier, 
    # but we must be careful not to double count in the objective?
    # The objective used upper triangle. 
    # If we use upper triangle in objective, the gradient should correspond to that.
    # However, the soft-min is symmetric. 
    # Let's stick to the objective definition: sum over i<j.
    
    # We need to map weights back to pairs.
    # i_upper, j_upper = np.triu_indices(N, k=1)
    # weights_pair = weights[:len(pair_vals)]
    
    # For each pair (i, j), the contribution to grad[i] is -weights[i,j] * (p_i - p_j) / (2 * d_ij)
    # And to grad[j] is -weights[i,j] * (p_j - p_i) / (2 * d_ij)
    
    # Let's compute this efficiently.
    # diff is p_i - p_j.
    # dists is |p_i - p_j|.
    # We need (p_i - p_j) / (2 * d_ij).
    # This is diff / (2 * dists).
    
    # We only have weights for upper triangle.
    # Let's create a full weight matrix W where W[i,j] = weight for pair (i,j) if i<j, 0 otherwise.
    # Actually, since the objective sums i<j, the gradient w.r.t p_i gets contribution from all j.
    
    # Let's just loop over the pairs in the upper triangle for gradient accumulation.
    # N=50, pairs=1225. Loop is fast enough.
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(pair_vals)]
    
    # Vectorized update for pairs
    # diff_ij = points[i_idx] - points[j_idx]
    # d_ij = dists[i_idx, j_idx]
    # term = w_pairs[:, np.newaxis] * diff_ij / (2 * d_ij[:, np.newaxis])
    # grad[i_idx] -= term
    # grad[j_idx] += term
    
    diff_ij = points[i_idx] - points[j_idx]
    d_ij = dists[i_idx, j_idx]
    # Avoid division by zero
    d_ij_safe = np.where(d_ij < 1e-8, 1e-8, d_ij)
    
    term = w_pairs[:, np.newaxis] * diff_ij / (2 * d_ij_safe[:, np.newaxis])
    
    # Accumulate into grad
    np.add.at(grad, i_idx, -term)
    np.add.at(grad, j_idx, term)
    
    # Boundary gradient
    # v = 1 - r_i. dv/dp_i = -p_i / r_i.
    # Contribution to grad: -weights_boundary * (-p_i / r_i) = weights_boundary * p_i / r_i
    w_bound = weights[len(pair_vals):]
    radii_safe = np.where(radii < 1e-8, 1e-8, radii)
    grad_bound = w_bound[:, np.newaxis] * points / radii_safe[:, np.newaxis]
    grad += grad_bound
    
    return obj, -grad # We want to minimize obj, which is log(sum)/beta.
    # Wait, earlier I said minimize obj = log(sum)/beta.
    # Let's re-verify signs.
    # We want to MAXIMIZE min(v).
    # Soft min approx: S = -1/beta log(sum(exp(-beta v))).
    # As beta -> inf, S -> min(v).
    # We want to maximize S.
    # So we minimize -S = 1/beta log(sum(exp(-beta v))).
    # So Objective to minimize is J = 1/beta log(sum(exp(-beta v))).
    # Gradient of J:
    # dJ/dx = 1/beta * 1/sum * sum(exp(-beta v) * (-beta dv/dx))
    # = -1/sum * sum(exp(-beta v) * dv/dx)
    # = - sum(weights * dv/dx)
    
    # In my code above:
    # I calculated `obj = np.log(sum_exp) / beta`. This is J.
    # I calculated `grad` as `- sum(weights * dv/dx)`?
    # Let's check the code:
    # `term = w_pairs ...`
    # `np.add.at(grad, i_idx, -term)` -> grad[i] -= w * (p_i-p_j)/(2d)
    # dv/dp_i for v=d/2 is (p_i-p_j)/(2d).
    # So grad[i] accumulates -w * dv/dp_i.
    # This matches `- sum(w * dv/dx)`.
    # So `grad` is the gradient of J.
    # We want to minimize J. So we return `grad`.
    
    # Wait, I returned `-grad` in the draft above. Let's correct that.
    # If `grad` is dJ/dx, and we want to minimize J, we return `grad`.
    
    # Let's re-read the draft return statement.
    # `return obj, -grad`
    # If `grad` variable holds `- sum(w dv/dx)`, which is `dJ/dx`, then returning `grad` is correct for minimization.
    # Let's trace:
    # J = 1/beta log(sum exp(-beta v))
    # dJ = 1/beta * 1/S * sum(exp(-beta v) * -beta dv)
    # dJ = -1/S * sum(exp(-beta v) dv)
    # dJ = - sum( (exp(-beta v)/S) dv )
    # dJ = - sum( weights dv )
    
    # My code:
    # `term = w_pairs ... * diff / (2d)` which is `w * dv/dp_i` for pair i.
    # `np.add.at(grad, i_idx, -term)` -> `grad[i] += -w * dv/dp_i`
    # So `grad` accumulates `- sum(w dv/dx)`.
    # So `grad` IS `dJ/dx`.
    # So we should return `grad`.
    
    return obj, grad

def _optimize_lbfgs(initial_points, beta_schedule):
    """Run L-BFGS-B with a schedule of beta values."""
    current_points = initial_points.copy()
    
    for beta in beta_schedule:
        res = minimize(
            lambda x: _objective_and_gradient(x, beta)[0],
            current_points.flatten(),
            jac=lambda x: _objective_and_gradient(x, beta)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        current_points = res.x.reshape(N, 2)
        
    return current_points

def _get_hex_lattice():
    """Generate initial points on a hexagonal lattice inside the unit disk."""
    points = []
    spacing = 0.2
    while len(points) < N:
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2.0)**2:
                    points.append([x, y])
        if len(points) >= N:
            break
        spacing *= 0.95
    points = np.array(points)
    radii = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(radii)
    return points[idx[:N]]

def _get_random_init():
    """Generate random initial points inside the unit disk."""
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0, 1.0, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _get_ring_init():
    """Generate points on concentric rings."""
    points = []
    points.append([0.0, 0.0])
    n_rings = 7
    for ring in range(1, n_rings):
        r = ring * 0.13
        n_pts = min(6 * ring, N - len(points))
        if n_pts <= 0:
            break
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    points = np.array(points[:N])
    return points

def _polish(points, beta_schedule):
    """Polish points with high beta values."""
    return _optimize_lbfgs(points, beta_schedule)

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Beta schedule: start low, increase to sharpen
    base_schedule = [10, 50, 100, 200, 500, 1000]
    
    initializers = [
        _get_hex_lattice,
        _get_ring_init,
        _get_random_init,
        _get_random_init,
        _get_random_init,
        _get_random_init,
        _get_random_init,
    ]
    
    for init_func in initializers:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        initial = init_func()
        # Run optimization with increasing beta
        points = _optimize_lbfgs(initial, base_schedule)
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
            
    # If we have time, do high-beta polishing on the best result
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 10.0 and best_points is not None:
        # High beta polish
        high_beta_schedule = [2000, 5000, 10000]
        polished_points = _polish(best_points, high_beta_schedule)
        polished_score = _score(polished_points)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    if best_points is None:
        best_points = _get_hex_lattice()
        
    return best_points
