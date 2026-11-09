import numpy as np
import time

N = 200

def compute_radius(points):
    if points is None:
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return float(r)

def hex_init(n_rows):
    """Generate a hexagonal lattice initialization with n_rows rows."""
    c_base = N // n_rows
    rem = N % n_rows
    counts = [c_base] * n_rows
    for i in range(rem):
        counts[i] += 1
    
    pts = []
    y = 0.0
    for i, c in enumerate(counts):
        xs = np.arange(c) - (c - 1) / 2.0
        if i % 2 == 1:
            xs += 0.5
        for x in xs:
            pts.append([x, y])
        y += np.sqrt(3) / 2.0
    
    pts = np.array(pts, dtype=np.float64)
    # Center and scale to fit in [-0.5, 0.5]
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    max_extent = max(np.max(np.abs(pts[:, 0])), np.max(np.abs(pts[:, 1])))
    if max_extent > 0:
        scale = 0.48 / max_extent
        pts *= scale
    return pts

def repulsive_relax(pts, n_steps=50, lr=0.001, force_k=1.0, wall_k=1.0):
    """Simple repulsive force-based relaxation to spread points apart."""
    pts = pts.copy()
    for step in range(n_steps):
        # Compute pairwise repulsive forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: k / dist^2 in direction of separation
        # Only consider pairs closer than some threshold
        force_mask = dist < 0.15
        force_mag = np.zeros_like(dist)
        force_mag[force_mask] = force_k / (dist[force_mask] ** 2)
        
        # Sum forces for each point
        forces = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion: push points away from walls
        x = pts[:, 0]
        y = pts[:, 1]
        wall_force_x = np.zeros_like(x)
        wall_force_y = np.zeros_like(y)
        
        # Left wall
        d_left = x + 0.5
        wall_force_x[d_left < 0.05] += wall_k * (0.05 - d_left[d_left < 0.05])
        # Right wall
        d_right = 0.5 - x
        wall_force_x[d_right < 0.05] -= wall_k * (0.05 - d_right[d_right < 0.05])
        # Bottom wall
        d_bottom = y + 0.5
        wall_force_y[d_bottom < 0.05] += wall_k * (0.05 - d_bottom[d_bottom < 0.05])
        # Top wall
        d_top = 0.5 - y
        wall_force_y[d_top < 0.05] -= wall_k * (0.05 - d_top[d_top < 0.05])
        
        forces[:, 0] += wall_force_x
        forces[:, 1] += wall_force_y
        
        # Update positions
        pts += lr * forces
        pts = np.clip(pts, -0.49, 0.49)
    
    return pts
    # points: (N, 2)
    # Calculate min pairwise distance
    # (N, 1, 2) - (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    # Zero out diagonal
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    # dist to walls: 0.5 - |x|, 0.5 - |y|
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return r

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(50):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # 2. Hexagonal
    # Try to fit hex grid
    for n_rows in range(10, 21):
        c_base = N // n_rows
        rem = N % n_rows
        counts = [c_base] * n_rows
        for i in range(rem):
            counts[i] += 1
            
        pts = []
        y = 0
        for i, c in enumerate(counts):
            xs = np.arange(c) - (c - 1) / 2.0
            if i % 2 == 1:
                xs += 0.5
            for x in xs:
                pts.append([x, y])
            y += np.sqrt(3) / 2.0
            
        pts = np.array(pts)
        # Normalize to fit in [-0.5, 0.5] roughly
        # Scale factor
        max_x = np.max(np.abs(pts[:, 0]))
        max_y = np.max(np.abs(pts[:, 1]))
        scale = 0.45 / max(max_x, max_y)
        pts *= scale
        inits.append(pts)
        
    # Optimization loop
    # We will use gradient ascent on soft-min
    # Objective to minimize: F = log(sum(exp(-lambda * dist_sq)))
    # Gradient descent on F
    
    lambda_val = 100.0 # High lambda to focus on min
    
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        # Simple gradient descent
        # We need to compute gradient of F
        # F = log(sum_pairs exp(-lambda d_ij^2)) + log(sum_walls exp(-lambda w_i^2))
        
        # To save time, we do a few steps of gradient descent
        # Then a few steps of random jitter
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            
            # Weights for pairs
            w_pairs = np.exp(-lambda_val * dist_sq)
            # Handle inf
            w_pairs[~np.isfinite(w_pairs)] = 0
            
            sum_w_pairs = np.sum(w_pairs, axis=1)
            
            # Gradient from pairs
            # grad_i = sum_j w_ij * (-2 lambda * (p_i - p_j)) / sum_w_i
            # Vectorized:
            # w_pairs is (N, N)
            # diff is (N, N, 2)
            # term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
            term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
            grad_pairs = np.sum(term, axis=1)
            
            # Normalize
            # Avoid division by zero
            sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
            grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
            
            # Wall terms
            x = pts[:, 0]
            y = pts[:, 1]
            
            # dist to left/right
            d_left = 0.5 + x # dist to -0.5? No. 0.5 - |x|.
            # Let's use explicit distances to 4 walls
            # w_x_left = 0.5 - (-x) = 0.5 + x ? No, x is in [-0.5, 0.5].
            # Distance to left wall (-0.5) is x - (-0.5) = x + 0.5
            # Distance to right wall (0.5) is 0.5 - x
            # Distance to bottom wall (-0.5) is y + 0.5
            # Distance to top wall (0.5) is 0.5 - y
            
            d_x_l = x + 0.5
            d_x_r = 0.5 - x
            d_y_b = y + 0.5
            d_y_t = 0.5 - y
            
            # Squared distances
            d_x_l_sq = d_x_l**2
            d_x_r_sq = d_x_r**2
            d_y_b_sq = d_y_b**2
            d_y_t_sq = d_y_t**2
            
            w_x_l = np.exp(-lambda_val * d_x_l_sq)
            w_x_r = np.exp(-lambda_val * d_x_r_sq)
            w_y_b = np.exp(-lambda_val * d_y_b_sq)
            w_y_t = np.exp(-lambda_val * d_y_t_sq)
            
            sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
            
            # Gradient contribution from walls
            # d/dx ( -lambda * (x+0.5)^2 ) = -2 lambda (x+0.5)
            # d/dx ( -lambda * (0.5-x)^2 ) = -2 lambda (0.5-x) * (-1) = 2 lambda (0.5-x)
            
            grad_x_walls = (w_x_l * (-2 * lambda_val * d_x_l) + w_x_r * (2 * lambda_val * d_x_r)) / sum_w_walls
            grad_y_walls = (w_y_b * (-2 * lambda_val * d_y_b) + w_y_t * (2 * lambda_val * d_y_t)) / sum_w_walls
            
            grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
            
            total_grad = grad_pairs + grad_walls
            
            # We want to MINIMIZE F. So we move in direction of -grad.
            # But wait, F is log-sum-exp of exp(-lambda d^2).
            # If d is small, exp(-lambda d^2) is large. F is large.
            # We want to make d large -> make exp term small -> make F small.
            # So we do gradient descent on F.
            # Step: pts = pts - lr * grad_F
            
            # Adaptive learning rate
            # Start with small lr, decay
            lr = 0.001 / (1 + step * 0.001)
            
            pts_new = pts - lr * total_grad
            
            # Ensure points stay in bounds? 
            # The gradient should push them in, but numerical errors might push out.
            # Clip to [-0.5, 0.5]
            pts_new = np.clip(pts_new, -0.5, 0.5)
            
            # Check improvement?
            # For speed, we don't check every step.
            # But we should occasionally check if we are stuck.
            
            pts = pts_new
            
            # Every 100 steps, check radius
            if step % 100 == 0:
                r = compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_points = pts.copy()
                    
                # Random restart/jitter if stuck?
                # Add small noise
                if step % 200 == 0:
                    pts += np.random.randn(N, 2) * 0.01
                    pts = np.clip(pts, -0.5, 0.5)
                    
    # Final check
    if best_points is not None:
        r = compute_radius(best_points)
        if r > best_r:
            best_r = r
            # best_points already set
    
    if best_points is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
