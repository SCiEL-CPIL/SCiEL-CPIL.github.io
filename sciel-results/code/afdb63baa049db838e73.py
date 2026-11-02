import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1
    best_result = None
    
    # Sweep aspect ratios
    # W + H = 2
    # Try a range of widths
    ws = np.linspace(0.7, 1.3, 15)
    
    for W in ws:
        if time.time() > deadline:
            break
            
        H = 2.0 - W
        
        # Initialize with a 5x5 grid, take first 21
        side = 5
        # Create grid points
        xs_grid = np.array([(i + 0.5) / side * W for i in range(side) for j in range(side)])
        ys_grid = np.array([(j + 0.5) / side * H for i in range(side) for j in range(side)])
        
        xs = xs_grid[:N].copy()
        ys = ys_grid[:N].copy()
        
        # Add jitter
        xs += np.random.normal(0, 0.02 * W, N)
        ys += np.random.normal(0, 0.02 * H, N)
        
        # Initial radii
        # Estimate max radius
        r_max = min(W, H) / side / 2.0
        rs = np.full(N, r_max * 0.8)
        
        # Local Search
        step_size = 0.05
        grow_step = 0.0005
        stall_count = 0
        max_stall = 200
        
        while time.time() < deadline - 5.0:  # Reserve 5s for other ratios/verification
            # Compute overlaps
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            req = rs[:, None] + rs[None, :]
            overlap = np.maximum(0, req - dist)
            
            if np.sum(overlap) > 1e-7:
                # Push apart
                inv_dist = 1.0 / np.where(dist > 1e-9, dist, 1e-9)
                
                fx = overlap * inv_dist * dx
                fy = overlap * inv_dist * dy
                
                total_fx = np.sum(fx, axis=1)
                total_fy = np.sum(fy, axis=1)
                
                xs += total_fx * step_size
                ys += total_fy * step_size
                
                # Keep inside box (soft constraint, hard clip later)
                # Don't clip here to allow pushing out of corners temporarily?
                # No, clipping prevents them from going far out.
                xs = np.clip(xs, 0, W)
                ys = np.clip(ys, 0, H)
                
                stall_count = 0
            else:
                # No overlaps, try to grow
                old_rs = rs.copy()
                rs += grow_step
                
                # Check if growth caused wall violations or overlaps
                # Wall violations
                if np.any(xs < rs) or np.any(xs > W - rs) or np.any(ys < rs) or np.any(ys > H - rs):
                    # Revert growth
                    rs = old_rs
                    # Perturb
                    xs += np.random.normal(0, 0.05 * W, N)
                    ys += np.random.normal(0, 0.05 * H, N)
                    xs = np.clip(xs, 0, W)
                    ys = np.clip(ys, 0, H)
                    stall_count += 1
                else:
                    # Check overlaps again (growth might cause overlap)
                    dx = xs[:, None] - xs[None, :]
                    dy = ys[:, None] - ys[None, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    np.fill_diagonal(dist, np.inf)
                    req = rs[:, None] + rs[None, :]
                    overlap = np.maximum(0, req - dist)
                    
                    if np.sum(overlap) > 1e-7:
                        # Overlap caused by growth, revert
                        rs = old_rs
                        stall_count += 1
                        # Perturb
                        xs += np.random.normal(0, 0.05 * W, N)
                        ys += np.random.normal(0, 0.05 * H, N)
                        xs = np.clip(xs, 0, W)
                        ys = np.clip(ys, 0, H)
                    else:
                        stall_count = 0
                        
                if stall_count > max_stall:
                    # Stronger perturbation
                    xs += np.random.normal(0, 0.1 * W, N)
                    ys += np.random.normal(0, 0.1 * H, N)
                    xs = np.clip(xs, 0, W)
                    ys = np.clip(ys, 0, H)
                    stall_count = 0
                    max_stall = 500 # Reset stall limit

        # Verification and Repair
        # Ensure strict validity
        # 1. Check walls
        # 2. Check overlaps
        
        # Repair walls
        for i in range(N):
            if xs[i] < rs[i]: xs[i] = rs[i]
            if xs[i] > W - rs[i]: xs[i] = W - rs[i]
            if ys[i] < rs[i]: ys[i] = rs[i]
            if ys[i] > H - rs[i]: ys[i] = H - rs[i]
            
        # Repair overlaps by shrinking radii
        # Simple approach: shrink all radii by a factor until no overlaps
        # This is conservative but ensures validity
        shrink_factor = 1.0
        for _ in range(100):
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            req = rs * shrink_factor
            req = req[:, None] + req[None, :]
            
            if np.all(dist >= req - 1e-9):
                break
            shrink_factor *= 0.95
            
        final_rs = rs * shrink_factor
        
        # Final check
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        req = final_rs[:, None] + final_rs[None, :]
        
        if np.all(dist >= req - 1e-9) and \
           np.all(xs >= final_rs - 1e-9) and np.all(xs <= W - final_rs + 1e-9) and \
           np.all(ys >= final_rs - 1e-9) and np.all(ys <= H - final_rs + 1e-9):
            
            score = np.sum(final_rs)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((xs, ys, final_rs))
        
    if best_result is None:
        # Fallback: Grid
        W = 1.0
        H = 1.0
        side = 5
        xs = np.array([(i + 0.5) / side * W for i in range(side) for j in range(side)][:N])
        ys = np.array([(j + 0.5) / side * H for i in range(side) for j in range(side)][:N])
        r = 0.5 / side * 0.9
        rs = np.full(N, r)
        best_result = np.column_stack((xs, ys, rs))
        
    return best_result
