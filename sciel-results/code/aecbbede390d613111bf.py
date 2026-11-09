import numpy as np
import time

N = 200

def compute_radius(points):
    """
    Calculate the radius of the largest equal circles that can be placed
    at the given centers within the square [-0.5, 0.5]^2.
    """
    if points is None or not np.all(np.isfinite(points)):
        return 0.0
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2):
        return 0.0
    
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    min_pair_dist = np.sqrt(min_pair_dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    # The radius is half the min pairwise distance, but also limited by wall distance
    r = np.min([min_pair_dist / 2.0, min_wall_dist])
    
    # Ensure non-negative
    if r < 0:
        return 0.0
    return r

def hex_init(rows):
    """
    Generate a hexagonal grid initialization with a specific number of rows.
    Returns an array of shape (N, 2) scaled to fit in [-0.5, 0.5].
    """
    if rows <= 0:
        return np.random.rand(N, 2) - 0.5
    
    # Estimate columns needed
    # In a hex grid, rows alternate between having k and k+1 points (or similar)
    # Let's generate a grid of rows x cols and trim/pad to N
    # To get a good aspect ratio, cols should be approx rows * 2/sqrt(3) for a square
    # But we just need N points.
    
    # Let's determine cols such that rows * cols >= N and aspect ratio is reasonable
    # Approximate points per row: N / rows
    cols = int(np.ceil(N / rows))
    
    # Generate raw hex grid points
    # Spacing d = 1 for now, we will scale later
    d = 1.0
    dy = np.sqrt(3) / 2.0 * d
    
    xs = []
    ys = []
    for r in range(rows):
        # Offset every other row
        offset = 0.5 * d if r % 2 == 1 else 0.0
        # Number of points in this row
        # We want to fill up to N points total
        # Let's just generate cols points in each row, then trim
        for c in range(cols):
            x = c * d + offset
            y = r * dy
            xs.append(x)
            ys.append(y)
            
    pts = np.column_stack((xs, ys))
    
    # Trim or pad to N
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with random points near the center of the existing cluster
        # Or just repeat? No, that causes overlap.
        # Let's add points randomly in the bounding box
        while len(pts) < N:
            x = np.random.uniform(np.min(pts[:,0]), np.max(pts[:,0]))
            y = np.random.uniform(np.min(pts[:,1]), np.max(pts[:,1]))
            pts = np.vstack([pts, np.array([x, y])])
            
    # Center the points
    center = np.mean(pts, axis=0)
    pts -= center
    
    # Scale to fit in [-0.5, 0.5] with a small margin
    # Find max extent
    max_extent = np.max(np.abs(pts))
    if max_extent > 0:
        # Scale so that the furthest point is at 0.45 (leaving room for movement)
        scale = 0.45 / max_extent
        pts *= scale
        
    return pts

def relax_forces(pts, steps=50, lr=0.01):
    """
    Apply repulsive force-based relaxation to push points apart.
    """
    for _ in range(steps):
        # Compute pairwise differences and distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, will be masked
        
        # Repulsive force: proportional to 1/d^2 (Coulomb-like)
        # Force on i from j is F_ij = (p_i - p_j) / d_ij^3 ? 
        # Gradient of 1/d is -1/d^2 * (p_i-p_j)/d. 
        # We want to maximize d, so we minimize -1/d. 
        # Grad of -1/d w.r.t p_i is (p_i - p_j) / d^3.
        # So we move p_i in direction (p_i - p_j) / d^3.
        
        # dist = sqrt(dist_sq)
        dist = np.sqrt(dist_sq)
        
        # Force vector
        # F_i = sum_j (p_i - p_j) / d_ij^3
        # To avoid overflow for very small d, clip d
        dist_safe = np.clip(dist, 1e-6, None)
        
        # Inverse cube distance
        inv_d3 = 1.0 / (dist_safe**3)
        inv_d3[~np.isfinite(inv_d3)] = 0
        np.fill_diagonal(inv_d3, 0)
        
        # Force is sum over j of diff_ij * inv_d3_ij
        # diff is (N, N, 2)
        # inv_d3 is (N, N)
        force = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces: Push inwards if close to walls
        # Distance to walls
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_x_l = x + 0.5
        d_x_r = 0.5 - x
        d_y_b = y + 0.5
        d_y_t = 0.5 - y
        
        # Force is strong if distance is small
        # F_x = (1/d_x_l^2) if close to left, etc.
        # Let's use a simple potential: U = 1/d^2. Grad is -2/d^3.
        # We want to move away from wall, so force is +1/d^3 inwards?
        # If x is close to -0.5, d_x_l is small. Force should be +x direction.
        # Grad of 1/d_x_l^2 w.r.t x is -2/d_x_l^3 * (1) (since d_x_l = x+0.5, d/dx=1).
        # We want to minimize U, so move in -grad direction?
        # No, we want to maximize distance. So we move in direction of increasing d.
        # Increasing d_x_l means increasing x. So force is +1/d_x_l^3.
        
        d_x_l_safe = np.clip(d_x_l, 1e-6, None)
        d_x_r_safe = np.clip(d_x_r, 1e-6, None)
        d_y_b_safe = np.clip(d_y_b, 1e-6, None)
        d_y_t_safe = np.clip(d_y_t, 1e-6, None)
        
        f_x_walls = 1.0/(d_x_l_safe**3) - 1.0/(d_x_r_safe**3)
        f_y_walls = 1.0/(d_y_b_safe**3) - 1.0/(d_y_t_safe**3)
        
        force_walls = np.column_stack((f_x_walls, f_y_walls))
        
        # Total force
        total_force = force + force_walls
        
        # Update
        pts += lr * total_force
        
        # Clip to bounds
        pts = np.clip(pts, -0.5, 0.5)
        
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    best_r = 0
    best_points = None
    
    inits = []
    
    # 1. Hexagonal inits with different aspect ratios
    # N=200. sqrt(200) ~ 14.
    for rows in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
        pts = hex_init(rows)
        inits.append(pts)
        
    # 2. Random inits
    for _ in range(5):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
        
    # Optimization
    for init in inits:
        if time.time() > deadline:
            break
            
        pts = init.copy()
        
        # Phase 1: Force relaxation
        # This helps settle the initial grid into a more natural packing
        # Use a decaying learning rate
        steps_relax = 100
        for i in range(steps_relax):
            if time.time() > deadline:
                break
            lr = 0.01 * (1.0 - i/steps_relax) + 0.001
            pts = relax_forces(pts, steps=1, lr=lr)
            
        # Check radius after relaxation
        r = compute_radius(pts)
        if r > best_r:
            best_r = r
            best_points = pts.copy()
            
        # Phase 2: Gradient descent on soft-min
        # F = -1/lambda * log(sum(exp(-lambda d^2)))
        # We want to maximize min d, so minimize F.
        
        lambda_schedule = [100, 200, 500, 1000]
        
        for lambda_val in lambda_schedule:
            if time.time() > deadline:
                break
                
            # Determine steps
            if lambda_val < 500:
                steps = 200
                lr = 0.005
            else:
                steps = 100
                lr = 0.001
                
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Compute distances
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=-1)
                np.fill_diagonal(dist_sq, np.inf)
                
                # Weights for pairs: exp(-lambda * dist_sq)
                w_pairs = np.exp(-lambda_val * dist_sq)
                w_pairs[~np.isfinite(w_pairs)] = 0
                
                sum_w_pairs = np.sum(w_pairs, axis=1)
                
                # Gradient from pairs
                term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
                grad_pairs = np.sum(term, axis=1)
                
                sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
                grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
                
                # Wall terms
                x = pts[:, 0]
                y = pts[:, 1]
                
                d_x_l = x + 0.5
                d_x_r = 0.5 - x
                d_y_b = y + 0.5
                d_y_t = 0.5 - y
                
                w_x_l = np.exp(-lambda_val * d_x_l**2)
                w_x_r = np.exp(-lambda_val * d_x_r**2)
                w_y_b = np.exp(-lambda_val * d_y_b**2)
                w_y_t = np.exp(-lambda_val * d_y_t**2)
                
                sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
                
                grad_x_walls = (w_x_l * (-2 * lambda_val * d_x_l) + w_x_r * (2 * lambda_val * d_x_r)) / sum_w_walls
                grad_y_walls = (w_y_b * (-2 * lambda_val * d_y_b) + w_y_t * (2 * lambda_val * d_y_t)) / sum_w_walls
                
                grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
                
                total_grad = grad_pairs + grad_walls
                
                # Update
                pts_new = pts - lr * total_grad
                
                # Clip to bounds
                pts_new = np.clip(pts_new, -0.5, 0.5)
                
                pts = pts_new
                
                # Occasionally check radius
                if step % 50 == 0:
                    r = compute_radius(pts)
                    if r > best_r:
                        best_r = r
                        best_points = pts.copy()
                        
    # Final polish
    if best_points is not None and time.time() < deadline - 1:
        pts = best_points.copy()
        lambda_val = 5000
        lr = 0.0001
        for _ in range(50):
            if time.time() > deadline:
                break
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            
            w_pairs = np.exp(-lambda_val * dist_sq)
            w_pairs[~np.isfinite(w_pairs)] = 0
            sum_w_pairs = np.sum(w_pairs, axis=1)
            
            term = w_pairs[:, :, np.newaxis] * (-2 * lambda_val) * diff
            grad_pairs = np.sum(term, axis=1)
            sum_w_pairs_safe = np.where(sum_w_pairs > 0, sum_w_pairs, 1.0)
            grad_pairs /= sum_w_pairs_safe[:, np.newaxis]
            
            x = pts[:, 0]
            y = pts[:, 1]
            d_x_l = x + 0.5
            d_x_r = 0.5 - x
            d_y_b = y + 0.5
            d_y_t = 0.5 - y
            
            w_x_l = np.exp(-lambda_val * d_x_l**2)
            w_x_r = np.exp(-lambda_val * d_x_r**2)
            w_y_b = np.exp(-lambda_val * d_y_b**2)
            w_y_t = np.exp(-lambda_val * d_y_t**2)
            
            sum_w_walls = w_x_l + w_x_r + w_y_b + w_y_t
            
            grad_x_walls = (w_x_l * (-2 * lambda_val * d_x_l) + w_x_r * (2 * lambda_val * d_x_r)) / sum_w_walls
            grad_y_walls = (w_y_b * (-2 * lambda_val * d_y_b) + w_y_t * (2 * lambda_val * d_y_t)) / sum_w_walls
            
            grad_walls = np.column_stack((grad_x_walls, grad_y_walls))
            
            total_grad = grad_pairs + grad_walls
            
            pts_new = pts - lr * total_grad
            pts_new = np.clip(pts_new, -0.5, 0.5)
            pts = pts_new
            
        r = compute_radius(pts)
        if r > best_r:
            best_points = pts
    else:
        # Fallback to grid if no points found
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points
