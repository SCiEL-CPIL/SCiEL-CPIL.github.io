import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    # Create grid points
    xs = np.array([(i + 0.5) / side for i in range(side)])
    ys = np.array([(j + 0.5) / side for j in range(side)])
    
    # Flatten grid to get N points
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Take first N points
    x0 = grid_x[:N]
    y0 = grid_y[:N]
    
    # Initial radii
    # Max radius in grid is 0.5/side. Let's start slightly smaller.
    r0 = np.full(N, 0.5 / side - 1e-3)
    
    # Flatten to a single vector for optimizer
    # Vector format: [x0, y0, r0, x1, y1, r1, ...]
    v0 = np.zeros(3 * N)
    for i in range(N):
        v0[3*i] = x0[i]
        v0[3*i+1] = y0[i]
        v0[3*i+2] = r0[i]
        
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Penalty weight
    C = 1000.0
    
    def objective(v):
        # Unpack
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Sum of radii (to be maximized, so we minimize negative)
        sum_r = np.sum(rs)
        
        # Overlap penalties
        pen = 0.0
        # Precompute distances? For N=26, O(N^2) is fine.
        # d_ij^2 = (xi-xj)^2 + (yi-yj)^2
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if d < min_dist:
                    overlap = min_dist - d
                    pen += overlap * overlap
        
        # Boundary penalties
        for i in range(N):
            # Distance to boundary
            db = min(xs[i], 1.0 - xs[i], ys[i], 1.0 - ys[i])
            if rs[i] > db:
                overflow = rs[i] - db
                pen += overflow * overflow
                
        return -sum_r + C * pen

    # Run optimization
    # L-BFGS-B is good for bounded problems
    # We can try to run it once. If time permits, maybe restart?
    # For N=26, one run is usually fast enough.
    
    res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Check time
    if time.time() > deadline:
        # Fallback to grid if optimization failed or took too long?
        # But we already passed the deadline check before starting.
        # If it finishes, we return the result.
        pass
        
    # Extract best solution
    v_best = res.x
    
    # Verify validity and calculate score
    # We should ensure the result is valid.
    # The optimizer might not perfectly satisfy constraints due to numerical errors.
    # We can project or just return if valid.
    # If invalid, we might need to shrink radii slightly.
    
    xs_best = v_best[0::3]
    ys_best = v_best[1::3]
    rs_best = v_best[2::3]
    
    # Check validity
    valid = True
    for i in range(N):
        db = min(xs_best[i], 1.0 - xs_best[i], ys_best[i], 1.0 - ys_best[i])
        if rs_best[i] > db:
            valid = False
            break
        for j in range(i+1, N):
            dx = xs_best[i] - xs_best[j]
            dy = ys_best[i] - ys_best[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < rs_best[i] + rs_best[j]:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, try to fix by shrinking radii or falling back to grid
        # A simple fix: scale radii down until valid? Or just return grid?
        # Given the high penalty, it's likely valid.
        # Let's try a simple repair: reduce radii by a small factor if invalid.
        # Or just return the grid if this fails, which is valid.
        # But let's assume it's valid or close.
        # To be safe, if invalid, we can reduce radii by 1% and recheck?
        # Or just return the grid result which is guaranteed valid.
        # However, the grid result is score 1.0. We want to beat it.
        # If the optimizer result is invalid, it scores 0.
        # So we must ensure validity.
        
        # Simple repair: shrink all radii by 5%
        rs_best *= 0.95
        # Re-check
        valid = True
        for i in range(N):
            db = min(xs_best[i], 1.0 - xs_best[i], ys_best[i], 1.0 - ys_best[i])
            if rs_best[i] > db:
                valid = False
                break
            for j in range(i+1, N):
                dx = xs_best[i] - xs_best[j]
                dy = ys_best[i] - ys_best[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < rs_best[i] + rs_best[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
             # Fallback to grid
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    # Construct output array
    out = np.zeros((N, 3))
    for i in range(N):
        out[i, 0] = xs_best[i]
        out[i, 1] = ys_best[i]
        out[i, 2] = rs_best[i]
        
    return out
