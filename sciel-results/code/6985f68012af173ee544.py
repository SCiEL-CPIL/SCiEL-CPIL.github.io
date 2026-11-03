import numpy as np
import time

N = 200
DEADLINE = 145.0  # Leave 5s margin for safety

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    n = len(pts)
    if n == 0:
        return 0.0
    
    # Boundary distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_boundary = np.minimum(dx, dy).min()
    
    if r_boundary < 0:
        return 0.0 # Invalid, outside square

    # Pairwise distances
    # Use chunking to avoid O(N^2) memory blowup if N was larger, 
    # but for N=200, N^2=40000 floats is fine (320KB). 
    # We can just compute the full matrix.
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    
    return min(min_pair / 2.0, r_boundary)

def _soft_min_value_and_grad(pts, beta):
    """
    Compute the value and gradient of the soft-min objective.
    Objective: f = softmin(min_pair_dist, min_boundary_dist)
    We maximize f.
    
    Let D be the vector of all pairwise distances (i<j) and boundary distances.
    f = -1/beta * log(sum(exp(-beta * D)))
    grad f = sum_j (exp(-beta * D_j) / sum(exp(-beta * D_k))) * grad D_j
    """
    n = len(pts)
    
    # 1. Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle for unique pairs, but for gradient 
    # it's easier to work with the full matrix and divide by 2 or handle symmetry.
    # Let's stick to the full matrix logic for simplicity, but be careful with weights.
    # Actually, the objective is min over ALL pairs. 
    # The soft min over the set of pairs {d_ij | i<j} is equivalent to 
    # soft min over the full matrix if we handle the diagonal correctly.
    
    # Boundary distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_b = np.minimum(dx, dy)
    
    # Combine distances
    # We treat each pair (i,j) i<j and each boundary i as a separate constraint.
    # To vectorize efficiently, we can compute weights for the full distance matrix 
    # and the boundary vector.
    
    # Calculate weights for pairwise
    neg_beta_dists = -beta * dists
    # Subtract max for stability
    max_neg_pair = np.max(neg_beta_dists)
    exp_pair = np.exp(neg_beta_dists - max_neg_pair)
    exp_pair[~np.isfinite(exp_pair)] = 0 # Handle inf on diagonal
    
    # Calculate weights for boundary
    neg_beta_b = -beta * r_b
    max_neg_b = np.max(neg_beta_b)
    exp_b = np.exp(neg_beta_b - max_neg_b)
    
    # Normalize weights globally
    sum_exp_pair = np.sum(exp_pair)
    sum_exp_b = np.sum(exp_b)
    total_sum = sum_exp_pair + sum_exp_b
    
    if total_sum == 0:
        return 0, np.zeros_like(pts)
        
    w_pair = exp_pair / total_sum
    w_b = exp_b / total_sum
    
    # Value
    # log_sum_exp = max_val + log(sum(exp(x - max_val)))
    # We need to combine the maxes correctly.
    # f = -1/beta * log( sum(exp(-beta*D)) )
    # = -1/beta * log( exp(max_neg_pair)*sum_exp_pair + exp(max_neg_b)*sum_exp_b )
    # = -1/beta * ( max_global + log( exp(max_neg_pair-max_global)*sum_exp_pair + ... ) )
    
    max_global = max(max_neg_pair, max_neg_b)
    log_sum = np.log(np.exp(max_neg_pair - max_global) * sum_exp_pair + 
                     np.exp(max_neg_b - max_global) * sum_exp_b)
    value = -(max_global + log_sum) / beta
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Pairwise gradient contribution
    # d(d_ij)/d(p_i) = (p_i - p_j) / d_ij
    # d(d_ij)/d(p_j) = (p_j - p_i) / d_ij
    # grad_pair[i] = sum_j w_pair[i,j] * (p_i - p_j) / d_ij
    # Note: w_pair is symmetric.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term = w_pair / dists
    term[~np.isfinite(term)] = 0
    
    # grad_pair[i] = p_i * sum_j term[i,j] - sum_j term[i,j] * p_j
    sum_term = term.sum(axis=1)
    weighted_pts = term @ pts
    grad_pair = pts * sum_term[:, None] - weighted_pts
    
    # Boundary gradient contribution
    # d(r_bi)/d(p_i) depends on which boundary is active (x or y) and sign.
    # r_bi = min(0.5 - |x_i|, 0.5 - |y_i|)
    # If 0.5 - |x_i| <= 0.5 - |y_i| (i.e., |x_i| >= |y_i|), active is x.
    #   d/dx_i (0.5 - |x_i|) = -sign(x_i)
    #   d/dy_i = 0
    # Else active is y.
    
    abs_x = np.abs(pts[:, 0])
    abs_y = np.abs(pts[:, 1])
    mask_x = abs_x >= abs_y
    
    grad_b = np.zeros_like(pts)
    grad_b[mask_x, 0] = -w_b[mask_x] * np.sign(pts[mask_x, 0])
    grad_b[~mask_x, 1] = -w_b[~mask_x] * np.sign(pts[~mask_x, 1])
    
    grad = grad_pair + grad_b
    
    return value, grad

def _initial_hex_grid(offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal grid of N points centered in the square."""
    # Estimate spacing
    # Area per circle ~ 1/sqrt(N)
    # Hex packing density ~ 0.9069
    # Spacing s such that N * (sqrt(3)/2 * s^2) ~ 1
    # s ~ sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate grid
    points = []
    # Determine range
    max_idx = int(1.0 / (s * 0.866)) + 2 # 0.866 ~ sqrt(3)/2
    
    for row in range(-max_idx, max_idx + 1):
        y = row * s * np.sqrt(3) / 2 + offset_y
        if y < -0.5 or y > 0.5:
            continue
        x_offset = 0.5 * s if row % 2 != 0 else 0.0
        for col in range(-max_idx, max_idx + 1):
            x = col * s + x_offset + offset_x
            if x < -0.5 or x > 0.5:
                continue
            points.append([x, y])
            
    points = np.array(points)
    if len(points) == 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    if len(points) > N:
        # Randomly select N points? Or take center?
        # Taking center is better for symmetry.
        # Sort by distance from center
        dists_to_center = np.sum(points**2, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points or expand grid?
        # For simplicity, if we don't have enough, just pad with random
        # But usually hex grid with this spacing will have > N
        pass
        
    return points[:N]

def construct():
    start_time = time.time()
    best_pts = None
    best_r = 0.0
    
    # Multi-start with hex grids
    # Try a few different offsets to escape local minima
    offsets = [
        (0.0, 0.0),
        (0.01, 0.0),
        (0.0, 0.01),
        (0.01, 0.01),
        (-0.01, 0.0),
        (0.0, -0.01),
    ]
    
    # Also try a random initialization if time permits
    # But hex is usually better.
    
    for ox, oy in offsets:
        if time.time() > DEADLINE:
            break
            
        pts = _initial_hex_grid(ox, oy)
        if len(pts) < N:
            continue
            
        # Optimize this instance
        r, pts = _optimize_instance(pts, start_time)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    # If we have time left, polish the best result
    if best_pts is not None and time.time() < DEADLINE - 10:
        best_pts = _polish(best_pts, best_r, start_time)
        best_r = _compute_radius(best_pts)
        
    if best_pts is None:
        # Emergency fallback
        best_pts = _initial_hex_grid(0.0, 0.0)
        
    return best_pts

def _optimize_instance(pts, global_start):
    """Run gradient ascent for a fixed time or until convergence."""
    n_steps = 200
    beta = 50.0
    lr = 0.005
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() > global_start + 15: # Spend ~15s per start
            break
            
        # Adaptive beta
        # Increase beta to make soft-min sharper
        beta = 50.0 + step * 5.0
        
        val, grad = _soft_min_value_and_grad(pts, beta)
        
        # Normalize gradient to avoid huge steps
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            grad = grad / norm
            
        # Step
        pts_new = pts + lr * grad
        
        # Project to square
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r_new = _compute_radius(pts_new)
        
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # If no improvement, maybe reduce lr or break?
            # For now, just continue, but if many failures, break
            # Simple strategy: keep updating pts even if r doesn't improve immediately,
            # as we are climbing a soft landscape.
            pts = pts_new
            
    return best_r, best_pts

def _polish(pts, current_r, global_start):
    """Final polishing with higher beta and smaller steps."""
    beta = 200.0
    lr = 0.001
    best_pts = pts.copy()
    best_r = current_r
    
    # Run for remaining time
    while time.time() < global_start + DEADLINE - 2:
        val, grad = _soft_min_value_and_grad(pts, beta)
        
        norm = np.linalg.norm(grad)
        if norm < 1e-8:
            break
            
        grad = grad / norm
        
        pts_new = pts + lr * grad
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        r_new = _compute_radius(pts_new)
        
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # If stuck, try a smaller lr or break
            lr *= 0.9
            if lr < 1e-6:
                break
            pts = pts_new # Still move to explore
        
    return best_pts
