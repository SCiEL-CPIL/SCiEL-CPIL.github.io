import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _soft_min(values, temperature=0.01):
    """Compute soft minimum using log-sum-exp trick."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    min_val = np.min(values)
    exp_vals = np.exp(-(values - min_val) / temperature)
    sum_exp = np.sum(exp_vals)
    return min_val + temperature * np.log(sum_exp)

def _smooth_neg_score(centers_flat, temperature=0.01):
    """
    Smooth objective: negative of soft-min radius.
    Uses log-sum-exp for soft-min over all constraints.
    """
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 10.0
    
    # Boundary constraints: distance to each edge
    dist_x_left = centers[:, 0] + 0.5
    dist_x_right = 0.5 - centers[:, 0]
    dist_y_bottom = centers[:, 1] + 0.5
    dist_y_top = 0.5 - centers[:, 1]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Collect all constraint values
    all_constraints = np.concatenate([
        dist_x_left, dist_x_right, dist_y_bottom, dist_y_top,
        pair_dists[np.isfinite(pair_dists)]
    ])
    
    # Soft minimum
    r_soft = _soft_min(all_constraints, temperature)
    
    if r_soft <= 0:
        return 10.0
    return -r_soft

def _smooth_neg_score_grad(centers_flat, temperature=0.01):
    """
    Gradient of the smooth objective.
    """
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return np.zeros_like(centers_flat)
    
    # Boundary constraints
    dist_x_left = centers[:, 0] + 0.5
    dist_x_right = 0.5 - centers[:, 0]
    dist_y_bottom = centers[:, 1] + 0.5
    dist_y_top = 0.5 - centers[:, 1]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # All constraint values
    all_constraints = np.concatenate([
        dist_x_left, dist_x_right, dist_y_bottom, dist_y_top,
        pair_dists[np.isfinite(pair_dists)]
    ])
    
    # Soft min weights
    min_val = np.min(all_constraints)
    exp_vals = np.exp(-(all_constraints - min_val) / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of soft_min w.r.t. each constraint is -weight_i
    # d(r_soft)/d(constraint_i) = -weight_i
    
    # Build gradient array
    grad = np.zeros_like(centers_flat)
    
    # Boundary gradients
    # d(dist_x_left)/d(x_i) = 1, d(dist_x_right)/d(x_i) = -1
    # d(dist_y_bottom)/d(y_i) = 1, d(dist_y_top)/d(y_i) = -1
    
    # Weights for boundary constraints
    w_xl = weights[:N]
    w_xr = weights[N:2*N]
    w_yb = weights[2*N:3*N]
    w_yt = weights[3*N:4*N]
    
    # Gradient contribution from boundaries
    grad[:, 0] += -w_xl + w_xr  # d(-r_soft)/dx = -d(r_soft)/dx
    grad[:, 1] += -w_yb + w_yt
    
    # Pairwise gradient
    w_pairs = weights[4*N:]
    
    # For each pair (i,j), d(pair_dist_ij)/d(centers_i) = (centers_i - centers_j) / dist_ij
    # We need to accumulate contributions
    pair_idx = np.argwhere(np.isfinite(pair_dists))
    
    for idx in pair_idx:
        i, j = idx[0], idx[1]
        if i >= j:
            continue
        w = w_pairs[idx[0] * N + idx[1]] if False else 0.0
        
    # Simpler: compute gradient via automatic differentiation-like approach
    # d(r_soft)/d(centers_k) = sum_i weight_i * d(constraint_i)/d(centers_k)
    
    # Let's compute more carefully
    grad = np.zeros_like(centers_flat)
    
    # Boundary contributions
    # constraint 0..N-1: dist_x_left = x_i + 0.5, d/dx_i = 1
    grad[:, 0] += -w_xl
    # constraint N..2N-1: dist_x_right = 0.5 - x_i, d/dx_i = -1
    grad[:, 0] += w_xr
    # constraint 2N..3N-1: dist_y_bottom = y_i + 0.5, d/dy_i = 1
    grad[:, 1] += -w_yb
    # constraint 3N..4N-1: dist_y_top = 0.5 - y_i, d/dy_i = -1
    grad[:, 1] += w_yt
    
    # Pairwise contributions
    # For pair (i,j) with i<j, constraint value = dist(i,j)/2
    # d(dist(i,j)/2)/d(centers_i) = (centers_i - centers_j) / dist(i,j)
    # d(dist(i,j)/2)/d(centers_j) = (centers_j - centers_i) / dist(i,j)
    
    # We need to map pair weights back
    pair_mask = np.isfinite(pair_dists)
    pair_flat = pair_dists[pair_mask]
    w_pairs = weights[4*N:]
    
    for i in range(N):
        for j in range(i+1, N):
            w = w_pairs[i*N + j]
            d = dists[i,j]
            if d > 1e-12:
                diff_vec = (centers[i] - centers[j]) / d
                grad[2*i] += -w * diff_vec[0]
                grad[2*i+1] += -w * diff_vec[1]
                grad[2*j] += -w * (-diff_vec[0])
                grad[2*j+1] += -w * (-diff_vec[1])
    
    return grad

def _score_from_centers(centers):
    """Compute the common radius for a set of centers inside [-0.5, 0.5]^2."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_bound = np.min(dist_bound)
    
    if min_bound < 0:
        return 0.0

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    if min_pair < 0:
        return 0.0

    return min(min_bound, min_pair)

def _neg_score(centers_flat):
    """Objective function: negative of the radius."""
    centers = centers_flat.reshape(-1, 2)
    score = _score_from_centers(centers)
    if score <= 0:
        return 10.0
    return -score

def _generate_hex_initial(sx, sy, x_shift, x_global, y_global, row_lengths):
    """
    Generate centers from hex parameters.
    """
    n_rows = len(row_lengths)
    centers = []
    for i, length in enumerate(row_lengths):
        if length == 0:
            continue
        # y position
        y = y_global + (i - (n_rows - 1) / 2.0) * sy
        
        # x start position
        # Center the row of 'length' circles around x_global
        # Shift odd rows by x_shift
        shift = x_shift if i % 2 != 0 else 0.0
        x_start = x_global + shift - sx * (length - 1) / 2.0
        
        for j in range(length):
            x = x_start + j * sx
            centers.append([x, y])
            
    centers = np.array(centers)
    if len(centers) != N:
        return None
    return centers

def _generate_jittered_hex(row_lengths, sx, sy, x_shift, x_global, y_global, jitter_scale=0.01, seed=42):
    """
    Generate jittered hexagonal pattern for better initialization.
    """
    rng = np.random.RandomState(seed)
    n_rows = len(row_lengths)
    centers = []
    for i, length in enumerate(row_lengths):
        if length == 0:
            continue
        y = y_global + (i - (n_rows - 1) / 2.0) * sy
        shift = x_shift if i % 2 != 0 else 0.0
        x_start = x_global + shift - sx * (length - 1) / 2.0
        
        for j in range(length):
            x = x_start + j * sx
            # Add jitter
            x += rng.uniform(-jitter_scale, jitter_scale)
            y += rng.uniform(-jitter_scale, jitter_scale)
            centers.append([x, y])
            
    centers = np.array(centers)
    if len(centers) != N:
        return None
    return centers

def _optimize_hex_pattern(pattern, deadline):
    """
    Optimize a hexagonal pattern using jittered init + smooth L-BFGS-B.
    Returns (best_score, best_centers).
    """
    n_rows = len(pattern)
    max_len = max(pattern)
    
    # Estimate initial spacing
    sx_init = 0.14
    sy_init = sx_init * np.sqrt(3) / 2.0
    
    best_score = -np.inf
    best_centers = None
    
    # Try a few starting offsets with jitter
    starts = []
    for shift_val in [0.0, sx_init/2, -sx_init/2]:
        for gx in [-0.03, 0.0, 0.03]:
            for gy in [-0.03, 0.0, 0.03]:
                starts.append([sx_init, sy_init, shift_val, gx, gy])
    
    for idx, init_params in enumerate(starts):
        if time.time() > deadline - 30:
            break
            
        # Generate jittered hex pattern
        jitter_scale = 0.005
        seed = 42 + idx
        centers = _generate_jittered_hex(pattern, *init_params, jitter_scale=jitter_scale, seed=seed)
        if centers is None:
            continue
        
        # Check if initial is valid
        init_score = _score_from_centers(centers)
        if init_score > best_score:
            best_score = init_score
            best_centers = centers.copy()
        
        # Flatten centers for optimizer
        x0 = centers.flatten()
        
        # Optimize with smooth L-BFGS-B
        try:
            result = minimize(
                _smooth_neg_score, 
                x0, 
                jac=_smooth_neg_score_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 10.0: # Valid result
                opt_centers = result.x.reshape(-1, 2)
                opt_score = _score_from_centers(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers.copy()
                    
        except Exception:
            pass
            
    return best_score, best_centers

def _refine_free_points(centers, deadline):
    """
    Refine a set of centers using smooth L-BFGS-B then Nelder-Mead polish.
    Returns (best_score, best_centers).
    """
    best_score = _score_from_centers(centers)
    best_centers = centers.copy()
    x0 = centers.flatten()
    
    # First: L-BFGS-B with smooth objective
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _smooth_neg_score, 
                x0, 
                jac=_smooth_neg_score_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < 10.0:
                opt_centers = result.x.reshape(-1, 2)
                opt_score = _score_from_centers(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers.copy()
                    x0 = opt_centers.flatten()
        except Exception:
            pass
    
    # Second: Nelder-Mead polish on the exact (nonsmooth) objective
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _neg_score, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < 10.0:
                opt_centers = result.x.reshape(-1, 2)
                opt_score = _score_from_centers(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers.copy()
        except Exception:
            pass
            
    return best_score, best_centers

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = -np.inf
    best_centers = None
    
    # Define row patterns for hexagonal-like packing
    # Total circles = 50
    patterns = [
        [7, 7, 7, 7, 7, 7, 7, 1], # 8 rows
        [8, 7, 7, 7, 7, 7, 7],    # 7 rows
        [7, 8, 7, 8, 7, 8, 5],    # 7 rows
        [6, 7, 7, 7, 7, 7, 8],    # 7 rows
        [5, 7, 7, 7, 7, 7, 9],    # 7 rows
        [6, 8, 6, 8, 6, 8, 8],    # 7 rows (50)
        [7, 6, 7, 6, 7, 6, 7, 4], # 8 rows
        [8, 6, 8, 6, 8, 6, 8],    # 7 rows (50)
        [6, 6, 7, 7, 7, 7, 7, 3], # 8 rows
        [5, 6, 7, 7, 7, 7, 7, 4], # 8 rows
        [4, 7, 7, 7, 7, 7, 7, 4], # 8 rows
        [8, 8, 7, 7, 7, 7, 6],    # 7 rows
        [6, 7, 7, 7, 7, 7, 7, 2], # 8 rows
        [7, 7, 7, 7, 7, 7, 6, 2], # 8 rows
    ]
    
    # Filter valid patterns
    valid_patterns = [p for p in patterns if sum(p) == N]
    
    # First pass: optimize hex patterns with jittered initialization
    for pattern in valid_patterns:
        if time.time() > deadline - 60:
            break
            
        score, centers = _optimize_hex_pattern(pattern, deadline)
        if centers is not None and score > best_score:
            best_score = score
            best_centers = centers
    
    # Fallback if no valid centers found (should not happen)
    if best_centers is None:
        # Simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_score = _score_from_centers(best_centers)
    
    # Second pass: refine with smooth L-BFGS-B + Nelder-Mead polish
    if time.time() < deadline - 20:
        score, centers = _refine_free_points(best_centers, deadline)
        if centers is not None and score > best_score:
            best_score = score
            best_centers = centers
    
    return best_centers
