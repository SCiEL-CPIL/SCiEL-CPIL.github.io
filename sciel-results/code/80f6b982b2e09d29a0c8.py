"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _time_left(deadline, t_start):
    return deadline - (time.time() - t_start)

def _compute_radius(pts):
    """Compute the maximum common radius for given points."""
    n = len(pts)
    if n < 2:
        return 0.5 - np.max(np.abs(pts))
    
    # Pairwise distances - vectorized
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    min_bound = np.min([
        np.min(pts[:, 0] + 0.5),
        np.min(0.5 - pts[:, 0]),
        np.min(pts[:, 1] + 0.5),
        np.min(0.5 - pts[:, 1])
    ])
    
    return min(min_pair, min_bound)

def _smooth_neg_radius_and_grad(pts, t=20.0):
    """
    Compute a smooth approximation of negative min-radius and its gradient.
    Uses soft-min via inverse power sum for smoothness.
    """
    n = len(pts)
    if n < 2:
        r = 0.5 - np.max(np.abs(pts))
        grad = np.zeros_like(pts)
        return -r, grad
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Soft min of pairwise distances / 2
    # Using: softmin(x_i) ≈ (1/t) * log(sum(exp(-t*x_i)))
    # For numerical stability, subtract max
    pair_dists = dists / 2.0
    min_pair_val = np.min(pair_dists)
    
    # Compute soft min of pair distances
    exp_vals = np.exp(-t * (pair_dists - min_pair_val))
    soft_min_pair = min_pair_val + np.log(np.sum(exp_vals)) / t
    
    # Boundary distances
    b_left = pts[:, 0] + 0.5
    b_right = 0.5 - pts[:, 0]
    b_bottom = pts[:, 1] + 0.5
    b_top = 0.5 - pts[:, 1]
    
    all_bound = np.concatenate([b_left, b_right, b_bottom, b_top])
    min_bound_val = np.min(all_bound)
    
    exp_bound = np.exp(-t * (all_bound - min_bound_val))
    soft_min_bound = min_bound_val + np.log(np.sum(exp_bound)) / t
    
    # Overall soft min radius
    # Combine pair and boundary soft mins
    # We want softmin(soft_min_pair, soft_min_bound)
    # But that's nested. Instead, let's just take the min of the two soft mins
    # Actually, for the gradient approach, let's compute gradient of each
    
    # Gradient of soft_min_pair with respect to pts
    # d(soft_min_pair)/d(pts_i) involves the weights from the exponential
    # Let's compute it properly
    
    # For pairwise: soft_min = min_val + (1/t)*log(sum_{i<j} exp(-t*d_ij/2))
    # d(soft_min)/d(pts_k) = (1/t) * sum_{i<j} exp(-t*d_ij/2) * (-t/2) * d(d_ij)/d(pts_k) / sum_{i<j} exp(-t*d_ij/2)
    # d(d_ij)/d(pts_k) = 2*(pts_i - pts_j)/d_ij if k==i, -2*(pts_i-pts_j)/d_ij if k==j, 0 otherwise
    
    # This is getting complex. Let me use a simpler approach:
    # Just compute the gradient numerically or use a different formulation.
    
    # Alternative: use the fact that the subgradient of min is a convex combination
    # of the subgradients of the active constraints.
    
    # For now, let's compute the gradient more carefully.
    
    # Soft min of pair distances:
    # f_pair = (1/t) * log(sum_{i<j} exp(-t * d_ij / 2))
    # df_pair/d(pts_k) = (1/t) * [sum_{i<j} exp(-t*d_ij/2) * (-t/2) * dd_ij/d(pts_k)] / sum_{i<j} exp(-t*d_ij/2)
    #                  = -1/2 * [sum_{i<j} w_ij * dd_ij/d(pts_k)] / sum_{i<j} w_ij
    # where w_ij = exp(-t*d_ij/2)
    
    # dd_ij/d(pts_k) for k=i: (pts_i - pts_j)/d_ij
    # dd_ij/d(pts_k) for k=j: (pts_j - pts_i)/d_ij
    
    # Let's compute this efficiently
    # weights for each pair
    w = np.exp(-t * (dists - min_pair_val))  # shifted for stability
    # But we need to be careful: the actual soft min uses exp(-t*d/2)
    # Let me redo with proper formulation
    
    # Actually, let me just use a simpler smooth objective:
    # Minimize: sum of squared violations + penalty
    # Or use: -log(sum(exp(-t*r_i))) type objective
    
    # Let me use a different approach: 
    # Objective: minimize the sum of exp(-t * clearance_i) for all constraints
    # This is smooth and its gradient is well-defined
    
    # Clearances:
    # Pair: c_ij = d_ij - 2r (but we don't know r)
    # Actually, let's define the objective as:
    # F = sum_{i<j} exp(-t * d_ij) + sum_{i} sum_{walls} exp(-t * dist_to_wall)
    # Minimizing F encourages all distances to be large
    
    # Gradient of F:
    # dF/d(pts_k) = sum_{j != k} exp(-t*d_kj) * (-t) * (pts_k - pts_j)/d_kj 
    #              + boundary terms
    
    # Let's compute this
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if d < 1e-10:
                continue
            w_ij = np.exp(-t * d)
            diff = pts[i] - pts[j]
            grad[i] += -t * w_ij * diff / d
            grad[j] += t * w_ij * diff / d
    
    # Boundary contributions
    # Left wall: dist = pts[i,0] + 0.5
    grad[:, 0] += -t * np.exp(-t * (pts[:, 0] + 0.5))
    # Right wall: dist = 0.5 - pts[i,0]
    grad[:, 0] += t * np.exp(-t * (0.5 - pts[:, 0]))
    # Bottom wall: dist = pts[i,1] + 0.5
    grad[:, 1] += -t * np.exp(-t * (pts[:, 1] + 0.5))
    # Top wall: dist = 0.5 - pts[i,1]
    grad[:, 1] += t * np.exp(-t * (0.5 - pts[:, 1]))
    
    # The objective value (for monitoring)
    F = np.sum(np.exp(-t * dists)) + np.sum(np.exp(-t * (pts[:, 0] + 0.5))) + \
        np.sum(np.exp(-t * (0.5 - pts[:, 0]))) + np.sum(np.exp(-t * (pts[:, 1] + 0.5))) + \
        np.sum(np.exp(-t * (0.5 - pts[:, 1])))
    
    return F, grad

def _local_optimize(pts, deadline, t_start, max_iter=5000, lr_init=0.001):
    """
    Local optimization using gradient descent with adaptive learning rate.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Adaptive parameters
    lr = lr_init
    best_F = np.inf
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for it in range(max_iter):
        if _time_left(deadline, t_start) < 5.0:
            break
        
        # Compute gradient
        F, grad = _smooth_neg_radius_and_grad(pts, t=15.0)
        
        # Adaptive learning rate: reduce if F increases, increase if it decreases consistently
        if F > best_F * 1.01:
            lr *= 0.7
        elif F < best_F * 0.99:
            lr = min(lr * 1.1, 0.01)
        
        # Update
        pts_new = pts - lr * grad
        
        # Clip to valid region
        margin = 1e-9
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5 + margin, 0.5 - margin)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5 + margin, 0.5 - margin)
        
        # Check if valid
        r_new = _compute_radius(pts_new)
        
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
        
        # Accept if F decreased or r improved
        if F < best_F or r_new > best_r:
            pts = pts_new
            best_F = F
        else:
            # Reject, try smaller step
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_pts, best_r

def _hex_initial():
    """Create a hexagonal lattice initial configuration."""
    points = []
    
    # Hexagonal packing parameters
    # For 200 circles, use about 14-15 rows
    n_rows = 14
    base_cols = 14
    rem = 200 - n_rows * base_cols  # 200 - 196 = 4
    
    # Rows 0-3 get 15 circles, rows 4-13 get 14
    row_counts = []
    for i in range(n_rows):
        if i < rem:
            row_counts.append(base_cols + 1)
        else:
            row_counts.append(base_cols)
    
    # Compute spacing
    # Vertical spacing between rows
    dy = 1.0 / (n_rows + 0.5)
    # Horizontal spacing
    dx = 1.0 / (max(row_counts) + 0.5)
    
    # Adjust for hexagonal offset
    dy *= np.sqrt(3) / 2  # Hexagonal vertical spacing
    
    for row in range(n_rows):
        n_cols = row_counts[row]
        offset = 0 if row % 2 == 0 else dx / 2
        
        # Center the row
        total_width = (n_cols - 1) * dx
        x_start = -total_width / 2 + offset
        
        for col in range(n_cols):
            x = x_start + col * dx
            y = -0.5 + dy/2 + row * dy
            # Clip to valid region
            x = np.clip(x, -0.49, 0.49)
            y = np.clip(y, -0.49, 0.49)
            points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many or too few, adjust
    if len(points) > N:
        # Randomly select N points
        idx = np.random.choice(len(points), size=N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        while len(points) < N:
            points = np.vstack([points, np.random.uniform(-0.4, 0.4, size=(1, 2))])
        points = points[:N]
    
    return points

def _grid_initial():
    """Create a square grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        return grid[:N]
    else:
        # Pad
        while len(grid) < N:
            grid = np.vstack([grid, np.random.uniform(-0.4, 0.4, size=(1, 2))])
        return grid[:N]

def _random_initial(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(N, 2))

def construct():
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_config = None
    best_score = -1
    
    # Try multiple initializations
    initializers = [
        _hex_initial,
        _grid_initial,
        lambda: _random_initial(42),
        lambda: _random_initial(123),
        lambda: _random_initial(456),
    ]
    
    for init_idx, init_func in enumerate(initializers):
        if _time_left(deadline, t_start) < 10.0:
            break
        
        pts = init_func()
        r0 = _compute_radius(pts)
        
        if r0 > best_score:
            best_score = r0
            best_config = pts.copy()
        
        # Local optimization
        pts_opt, r_opt = _local_optimize(pts, deadline, t_start, max_iter=3000, lr_init=0.002)
        
        if r_opt > best_score:
            best_score = r_opt
            best_config = pts_opt.copy()
        
        # If time permits, try a second round with smaller learning rate
        if _time_left(deadline, t_start) > 15.0:
            pts_opt2, r_opt2 = _local_optimize(pts_opt, deadline, t_start, max_iter=2000, lr_init=0.0005)
            if r_opt2 > best_score:
                best_score = r_opt2
                best_config = pts_opt2.copy()
    
    # Final validation
    if best_config is None:
        best_config = _grid_initial()
    
    # Ensure all points are strictly inside
    margin = 1e-9
    best_config[:, 0] = np.clip(best_config[:, 0], -0.5 + margin, 0.5 - margin)
    best_config[:, 1] = np.clip(best_config[:, 1], -0.5 + margin, 0.5 - margin)
    
    return best_config

# EVOLVE-BLOCK-END
