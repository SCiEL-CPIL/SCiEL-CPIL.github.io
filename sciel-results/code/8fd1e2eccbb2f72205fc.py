import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    def hex_init():
        # Create a hexagonal grid close to the optimal density
        # Optimal r ~ 0.05315. Distance between centers ~ 2r ~ 0.1063.
        # Area of hex cell = s^2 * sqrt(3)/2. Density = 1 / (s^2 * sqrt(3)/2).
        # N ~ Area / CellArea => s ~ sqrt(2 * Area / (N * sqrt(3)))
        s_est = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        # Try a few spacings to ensure we can get N points inside the circle
        best_pts = None
        for s_mult in [0.95, 0.98, 1.0, 1.02, 1.05]:
            s = s_est * s_mult
            dx = s
            dy = s * np.sqrt(3) / 2
            
            # Generate grid points
            # Range needs to cover radius 1
            max_idx = int(1.0 / min(dx, dy)) + 2
            points = []
            for i in range(-max_idx, max_idx + 1):
                for j in range(-max_idx, max_idx + 1):
                    x = i * dx + (j % 2) * dx / 2
                    y = j * dy
                    if x*x + y*y <= 1.0:
                        points.append([x, y])
            
            if not points:
                continue
                
            points = np.array(points)
            
            if len(points) >= N:
                # Select N points. To maximize min distance, random selection is okay for init,
                # but we can try to keep the ones closest to the center to maintain symmetry?
                # Actually, for packing, boundary points are important. 
                # Random selection from a dense hex grid is a standard robust init.
                idx = np.random.choice(len(points), N, replace=False)
                pts = points[idx]
                if best_pts is None or len(pts) == N:
                    best_pts = pts
            else:
                # If too few, we might need a smaller s, but let's just use what we have
                # and let the optimizer push them out? No, we need N points.
                # If len(points) < N, this s is too large.
                pass
                
        if best_pts is None:
            # Fallback: Random uniform in disk
            r = np.sqrt(np.random.rand(N))
            theta = 2 * np.pi * np.random.rand(N)
            best_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
        return best_pts

    c = hex_init()
    
    # Ensure valid start: inside circle
    norms = np.linalg.norm(c, axis=1)
    mask_out = norms > 0.99
    if np.any(mask_out):
        c[mask_out] = c[mask_out] / norms[mask_out, None] * 0.99

    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Optimization parameters
    beta = 20.0  # Start higher for faster convergence to min gap
    lr = 0.05    # Higher learning rate to move faster
    num_steps = 20000 # Reduced steps, but higher LR
    best_score = -1.0
    best_c = c.copy()
    
    # Track time
    last_time_check = time.time()
    
    for step in range(num_steps):
        # Time check every 100 steps to avoid overhead
        if step % 100 == 0:
            current_time = time.time()
            if current_time - start_time > time_limit:
                break
            last_time_check = current_time
            
        # Compute distances
        diff = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Boundary
        norms = np.linalg.norm(c, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min objective: Maximize min(gaps)
        # Gaps: dists/2 and bound_dists
        # J = -1/beta * log(sum(exp(-beta * gap)))
        
        # To handle numerical stability, we can subtract the min gap before exp?
        # Or just rely on beta not being too huge initially.
        # With beta=20, exp(-20*0.05) = exp(-1) ~ 0.37. Safe.
        
        exp_pair = np.exp(-beta * dists / 2.0)
        exp_bound = np.exp(-beta * bound_dists)
        
        Z = np.sum(exp_pair) + np.sum(exp_bound)
        
        w_pair = exp_pair / Z
        w_bound = exp_bound / Z
        
        # Gradient calculation
        # dJ/dc_i = sum_k w_k * (dx_k/dc_i)
        
        # Pair gradients
        safe_dists = np.maximum(dists, 1e-9)
        factor = w_pair / (2.0 * safe_dists)
        diff_vec = c[i_idx] - c[j_idx]
        contrib = factor[:, None] * diff_vec
        
        # Using np.add.at is the standard way, but can be slow.
        # Alternative: Use bincount if possible? No, indices are arbitrary.
        # Let's stick to add.at but ensure we are as fast as possible.
        grad = np.zeros_like(c)
        
        np.add.at(grad, i_idx, contrib)
        np.add.at(grad, j_idx, -contrib)
        
        # Boundary gradients
        safe_norms = np.maximum(norms, 1e-9)
        bound_grad = w_bound[:, None] * (-c / safe_norms[:, None])
        grad += bound_grad
        
        # Update
        c += lr * grad
        
        # Project to disk
        new_norms = np.linalg.norm(c, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            c[mask] = c[mask] / new_norms[mask, None]
            
        # Calculate hard score
        min_pair_dist = np.min(dists) / 2.0
        min_bound_dist = np.min(bound_dists)
        curr_score = min(min_pair_dist, min_bound_dist)
        
        if curr_score > best_score:
            best_score = curr_score
            best_c = c.copy()
            
        # Annealing
        # Increase beta to focus on the minimum gap
        if step % 500 == 0:
            beta *= 1.2
            lr *= 0.95
            
    # Final Polish Phase
    # Use the best configuration found so far
    c = best_c.copy()
    beta = 100.0
    lr = 0.005
    
    # Run a few more steps with high beta to refine
    steps_polish = 1000
    for step in range(steps_polish):
        if time.time() - start_time > time_limit - 5: # Reserve 5s for return
            break
            
        diff = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        norms = np.linalg.norm(c, axis=1)
        bound_dists = 1.0 - norms
        
        exp_pair = np.exp(-beta * dists / 2.0)
        exp_bound = np.exp(-beta * bound_dists)
        
        Z = np.sum(exp_pair) + np.sum(exp_bound)
        
        w_pair = exp_pair / Z
        w_bound = exp_bound / Z
        
        safe_dists = np.maximum(dists, 1e-9)
        factor = w_pair / (2.0 * safe_dists)
        diff_vec = c[i_idx] - c[j_idx]
        contrib = factor[:, None] * diff_vec
        
        grad = np.zeros_like(c)
        np.add.at(grad, i_idx, contrib)
        np.add.at(grad, j_idx, -contrib)
        
        safe_norms = np.maximum(norms, 1e-9)
        bound_grad = w_bound[:, None] * (-c / safe_norms[:, None])
        grad += bound_grad
        
        c += lr * grad
        
        new_norms = np.linalg.norm(c, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            c[mask] = c[mask] / new_norms[mask, None]
            
        min_pair_dist = np.min(dists) / 2.0
        min_bound_dist = np.min(bound_dists)
        curr_score = min(min_pair_dist, min_bound_dist)
        
        if curr_score > best_score:
            best_score = curr_score
            best_c = c.copy()
            
    return best_c
