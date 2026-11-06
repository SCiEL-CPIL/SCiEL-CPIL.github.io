import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

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

def _optimize_hex_pattern(pattern, deadline):
    """
    Optimize a hexagonal pattern using Nelder-Mead on the 5 lattice parameters.
    Returns (best_score, best_centers).
    """
    n_rows = len(pattern)
    max_len = max(pattern)
    
    # Estimate initial spacing
    # Width is 1. If max_len circles, spacing sx ~ 1/(max_len-1) ?
    # But they must fit in [-0.5, 0.5].
    # Ideal hex: sx = 2*r, sy = sqrt(3)*r.
    # Let's guess r ~ 0.07. sx ~ 0.14, sy ~ 0.12.
    
    sx_init = 0.14
    sy_init = sx_init * np.sqrt(3) / 2.0
    
    best_score = -np.inf
    best_centers = None
    
    # Try a few starting offsets
    starts = []
    for shift_val in [0.0, sx_init/2, -sx_init/2]:
        for gx in [-0.05, 0.0, 0.05]:
            for gy in [-0.05, 0.0, 0.05]:
                starts.append([sx_init, sy_init, shift_val, gx, gy])
    
    for init_params in starts:
        if time.time() > deadline:
            break
            
        centers = _generate_hex_initial(*init_params, pattern)
        if centers is None:
            continue
        
        # Check if initial is valid
        init_score = _score_from_centers(centers)
        if init_score > best_score:
            best_score = init_score
            best_centers = centers.copy()
        
        # Flatten centers for optimizer
        x0 = centers.flatten()
        
        # Optimize
        try:
            # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness
            result = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
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
    Refine a set of centers using free-point optimization.
    Returns (best_score, best_centers).
    """
    best_score = _score_from_centers(centers)
    best_centers = centers.copy()
    x0 = centers.flatten()
    
    # Try a few optimization methods
    methods = [
        ('Nelder-Mead', {'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11}),
        ('Powell', {'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11}),
    ]
    
    for method, options in methods:
        if time.time() > deadline:
            break
        try:
            result = minimize(_neg_score, x0, method=method, options=options)
            if result.fun < 10.0:
                opt_centers = result.x.reshape(-1, 2)
                opt_score = _score_from_centers(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers.copy()
                    x0 = opt_centers.flatten()
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
        [7, 7, 7, 7, 7, 7, 7, 1], # 8 rows (duplicate, but different starting offset)
        [8, 8, 7, 7, 7, 7, 6],    # 7 rows
        [6, 7, 7, 7, 7, 7, 7, 2], # 8 rows
        [7, 7, 7, 7, 7, 7, 6, 2], # 8 rows
    ]
    
    # Filter valid patterns
    valid_patterns = [p for p in patterns if sum(p) == N]
    
    # First pass: optimize hex patterns
    for pattern in valid_patterns:
        if time.time() > deadline:
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
    
    # Second pass: refine with free points
    if time.time() < deadline:
        score, centers = _refine_free_points(best_centers, deadline)
        if centers is not None and score > best_score:
            best_score = score
            best_centers = centers
    
    return best_centers
