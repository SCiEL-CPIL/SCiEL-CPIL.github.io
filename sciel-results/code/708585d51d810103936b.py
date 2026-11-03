"""Improved packing: hexagonal lattice initialization + gradient-based optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _compute_radius(centres):
    """Compute the radius for a set of centres (must be inside [-0.5,0.5]^2)."""
    if len(centres) == 0:
        return 0.0
    # Boundary constraint
    dist_to_boundary = np.min(0.5 - np.abs(centres), axis=1)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(dist_to_boundary))


def _softmin_logsumexp(x, beta):
    """Compute soft minimum using log-sum-exp trick for stability."""
    # For negative values, softmin = -logsumexp(-x/beta) * beta
    neg_x = -x / beta
    max_val = np.max(neg_x)
    if max_val == -np.inf:
        return 0.0
    shifted = neg_x - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_val
    return -log_sum_exp * beta


def _grad_softmin_neg(x, beta):
    """Gradient of -softmin(x) = -(-logsumexp(-x/beta)*beta) = logsumexp(-x/beta)*beta"""
    # d/dx_i [softmin(x)] = exp(-x_i/beta) / sum_j exp(-x_j/beta)
    # So d/dx_i [-softmin(x)] = -exp(-x_i/beta) / sum_j exp(-x_j/beta)
    neg_x = -x / beta
    max_val = np.max(neg_x)
    shifted = neg_x - max_val
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    return -weights


def _negative_radius(centres, beta=0.01):
    """
    Compute negative of the radius using softmin surrogate.
    We want to minimize this function.
    
    r = min(min_pair_dist/2, min_boundary_dist)
    -r = -min(min_pair_dist/2, min_boundary_dist)
       = min(-min_pair_dist/2, -min_boundary_dist)
       = softmin of [-pair_dists/2, -boundary_dists]
    """
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Boundary distances
    boundary_dists = np.min(0.5 - np.abs(centres), axis=1)
    
    # Collect all constraints: -pair_dists and -boundary_dists
    # For softmin, we want min of these values
    # -pair_dists: shape (N, N) with inf on diagonal
    # -boundary_dists: shape (N,)
    
    # Reshape pair_dists to 1D, exclude diagonal
    pair_flat = pair_dists.ravel()
    # Mask out diagonal elements
    mask = ~np.eye(N, dtype=bool).ravel()
    pair_flat = pair_flat[mask]
    
    # Combine
    all_constraints = np.concatenate([-pair_flat, -boundary_dists])
    
    # Softmin
    return _softmin_logsumexp(all_constraints, beta)


def _grad_negative_radius(centres, beta=0.01):
    """Compute gradient of negative_radius with respect to centres."""
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Boundary distances
    boundary_dists = np.min(0.5 - np.abs(centres), axis=1)
    
    # We need d/dcentres [softmin(-pair_dists, -boundary_dists)]
    # softmin(x) = -logsumexp(-x/beta)*beta
    # d/dx_i softmin(x) = exp(-x_i/beta) / sum_j exp(-x_j/beta)
    
    # For pair distances:
    # d/dcentres_i [pair_dist(i,j)] = (centres_i - centres_j) / |centres_i - centres_j|
    # d/dcentres_i [-pair_dist(i,j)] = -(centres_i - centres_j) / |centres_i - centres_j|
    
    # For boundary distances:
    # boundary_dist(i) = min(0.5 - |c_ix|, 0.5 - |c_iy|)
    # This is tricky because of the min. Let's approximate with softmin too.
    
    # Actually, let's use a different approach: compute gradient numerically or use a simpler surrogate.
    
    # Let's use L-BFGS-B with finite differences for the gradient, or compute analytically.
    
    # For simplicity, let's compute the gradient of the softmin of all constraints.
    
    # All constraints: -pair_dists (flattened, no diagonal) and -boundary_dists
    pair_flat = pair_dists.ravel()
    mask = ~np.eye(N, dtype=bool).ravel()
    pair_flat = pair_flat[mask]
    
    all_constraints = np.concatenate([-pair_flat, -boundary_dists])
    
    # Weights for softmin
    neg_x = -all_constraints / beta
    max_val = np.max(neg_x)
    shifted = neg_x - max_val
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient with respect to constraints
    grad_constraints = weights  # d softmin / d constraint_i = weight_i
    
    # Now propagate to centres
    grad_centres = np.zeros_like(centres)
    
    # Pair constraints
    n_pairs = N * (N - 1) // 2
    pair_weights = grad_constraints[:2 * n_pairs]  # Wait, this is wrong.
    
    # Let me redo this. pair_flat has N*(N-1) elements (excluding diagonal)
    n_pair_elements = N * (N - 1)
    pair_weights = grad_constraints[:n_pair_elements]
    boundary_weights = grad_constraints[n_pair_elements:]
    
    # For pair (i, j), the constraint is -dists[i,j]/2
    # d/dcentres_k [-dists[i,j]/2] = ?
    
    # dists[i,j] = |c_i - c_j|
    # d/dc_i dists[i,j] = (c_i - c_j) / |c_i - c_j|
    # d/dc_j dists[i,j] = (c_j - c_i) / |c_j - c_i| = -(c_i - c_j) / |c_i - c_j|
    
    # So d/dc_i [-dists[i,j]/2] = -(c_i - c_j) / (2 * |c_i - c_j|)
    # And d/dc_j [-dists[i,j]/2] = (c_i - c_j) / (2 * |c_i - c_j|)
    
    # The pair_flat array is from dists.ravel(), which is in row-major order.
    # Element index k corresponds to (i, j) where i = k // N, j = k % N
    # But we excluded diagonal, so we need to map back.
    
    # Let's just iterate over pairs
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = pair_weights[idx]
            if w != 0:
                d = diff[i, j]
                dist = dists[i, j]
                if dist > 1e-10:
                    grad_centres[i] -= w * d / (2 * dist)
                    grad_centres[j] += w * d / (2 * dist)
            idx += 1
    
    # Boundary constraints
    # boundary_dists[i] = min(0.5 - |c_ix|, 0.5 - |c_iy|)
    # This is a min of two terms. Let's approximate the gradient.
    # For each i, we have weight boundary_weights[i]
    # d/dc_ix [0.5 - |c_ix|] = -sign(c_ix)
    # d/dc_iy [0.5 - |c_iy|] = -sign(c_iy)
    # But we have min, so the gradient is weighted by which term is smaller.
    # For simplicity, let's use the average or just use the sign of the coordinate.
    
    for i in range(N):
        w = boundary_weights[i]
        if w != 0:
            # Approximate: gradient is proportional to -sign(c_i)
            # More accurately, we should weight by which boundary is closer
            x_b = 0.5 - abs(centres[i, 0])
            y_b = 0.5 - abs(centres[i, 1])
            total = x_b + y_b
            if total > 1e-10:
                # Weight by which boundary is closer (smaller distance means more contribution to min)
                w_x = x_b / total
                w_y = y_b / total
                # Actually, for min, the gradient is 1 for the active constraint and 0 for others.
                # For softmin, we use weights.
                # d/dc_ix min(x_b, y_b) ≈ w_x * (-sign(c_ix)) + w_y * 0 (if x_b < y_b)
                # Let's just use the sign approximation
                grad_centres[i, 0] -= w * np.sign(centres[i, 0])
                grad_centres[i, 1] -= w * np.sign(centres[i, 1])
    
    return grad_centres


def _initial_hex_grid(n):
    """Generate initial hexagonal grid of n points in [-0.5, 0.5]^2."""
    # Hexagonal packing in a square
    # Spacing s such that we get approximately n points
    # Area of square = 1, area per point ≈ 1/n
    # Hex cell area = s^2 * sqrt(3)/2
    # So s ≈ sqrt(2/(n * sqrt(3)))
    
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    
    points = []
    # Generate hex lattice
    # Row spacing: s * sqrt(3) / 2
    row_spacing = s * np.sqrt(3) / 2
    
    # Number of rows
    n_rows = int(1.0 / row_spacing) + 2
    
    for i in range(-n_rows, n_rows + 1):
        y = i * row_spacing
        if y > 0.5 or y < -0.5:
            continue
        # Offset for odd rows
        offset = 0.0 if i % 2 == 0 else s / 2
        n_cols = int(1.0 / s) + 2
        for j in range(-n_cols, n_cols + 1):
            x = j * s + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many, select n points using greedy farthest-point
    if len(points) > n:
        # Greedy selection
        selected = [0]
        remaining = set(range(len(points)))
        remaining.discard(0)
        
        min_dists = np.full(len(points), np.inf)
        
        for _ in range(n - 1):
            new_pt = points[selected[-1]]
            d = np.sqrt(np.sum((points - new_pt) ** 2, axis=1))
            min_dists = np.minimum(min_dists, d)
            
            # Also consider boundary
            boundary_dists = np.min(0.5 - np.abs(points), axis=1)
            scores = np.minimum(min_dists, boundary_dists)
            
            for idx in selected:
                scores[idx] = -np.inf
            
            best_idx = int(np.argmax(scores))
            if best_idx not in remaining:
                for idx in remaining:
                    if scores[idx] > -np.inf:
                        best_idx = idx
                        break
                else:
                    break
            
            selected.append(best_idx)
            remaining.discard(best_idx)
        
        points = points[selected]
    elif len(points) < n:
        # Pad with random points
        while len(points) < n:
            pt = np.random.uniform(-0.5, 0.5, size=2)
            points = np.vstack([points, pt])
    
    return points[:n]


def construct():
    """Construct 200 circle centres in [-0.5, 0.5]^2 maximizing minimum radius."""
    t0 = time.time()
    
    # Initial hexagonal grid
    centres = _initial_hex_grid(N)
    
    # Ensure valid
    centres = np.clip(centres, -0.5, 0.5)
    
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Gradient-based optimization
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Flatten centres
    x0 = centres.ravel()
    
    def objective(x):
        c = x.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        return -_compute_radius(c)
    
    def gradient(x):
        c = x.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        return _grad_negative_radius(c, beta=0.005).ravel()
    
    # Try optimization
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < objective(x0):
            new_centres = result.x.reshape(N, 2)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_centres = new_centres
                best_r = new_r
        
        # Check time
        if time.time() - t0 > TIME_LIMIT - 10:
            return best_centres
        
        # Second round with smaller beta for sharper minimum
        x0 = best_centres.ravel()
        
        result2 = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result2.success or result2.fun < objective(x0):
            new_centres = result2.x.reshape(N, 2)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_centres = new_centres
                best_r = new_r
    
    except Exception as e:
        pass
    
    # Final local refinement with small perturbations
    if time.time() - t0 < TIME_LIMIT - 5:
        for _ in range(50):
            if time.time() - t0 > TIME_LIMIT - 2:
                break
            i = np.random.randint(N)
            orig = best_centres[i].copy()
            best_centres[i] = orig + np.random.randn(2) * 0.001
            best_centres[i] = np.clip(best_centres[i], -0.5, 0.5)
            
            new_r = _compute_radius(best_centres)
            if new_r > best_r:
                best_r = new_r
            else:
                best_centres[i] = orig
    
    return best_centres
