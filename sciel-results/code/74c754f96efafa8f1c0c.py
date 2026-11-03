import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: grid layout
    # Perimeter 4 -> w+h=2. Let's try w=1, h=1 initially.
    w0 = 1.0
    h0 = 2.0 - w0
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    rs = []
    for i in range(side):
        for j in range(side):
            if len(xs) >= N:
                break
            # Center of cell
            cx = (i + 0.5) / side * w0
            cy = (j + 0.5) / side * h0
            r = min(cx, w0 - cx, cy, h0 - cy) * 0.9
            xs.append(cx)
            ys.append(cy)
            rs.append(r)
    
    # Pad if necessary (should not happen for N=21 with side=5)
    while len(xs) < N:
        xs.append(w0/2)
        ys.append(h0/2)
        rs.append(0.1)
        
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    rs = np.array(rs[:N])
    
    # Variable vector: [w, x1..xN, y1..yN, r1..rN]
    x0 = np.concatenate(([w0], xs, ys, rs))
    
    def objective(v):
        w = v[0]
        if w <= 0 or w >= 2:
            return 1e10
        h = 2.0 - w
        
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        # Penalty for negative radii
        if np.any(rs < 0):
            return 1e10
            
        # Boundary penalties
        # Circles must be inside [0, w] x [0, h]
        # x_i >= r_i, x_i <= w - r_i, etc.
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        # Overlap penalties
        # d_ij >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d:
                    pen += (min_d - d)**2
        
        # We want to maximize sum(rs), so minimize -sum(rs)
        return -np.sum(rs) + 100.0 * pen

    # Gradient approximation might be slow, use numerical or L-BFGS-B with finite differences
    # L-BFGS-B is robust
    
    best_val = 1e10
    best_x = x0.copy()
    
    # Try a few different initial aspect ratios
    for w_init in [1.0, 0.8, 1.2, 0.5, 1.5]:
        if time.time() > deadline:
            break
            
        h_init = 2.0 - w_init
        # Re-initialize grid for this aspect ratio
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                cx = (i + 0.5) / side * w_init
                cy = (j + 0.5) / side * h_init
                r = min(cx, w_init - cx, cy, h_init - cy) * 0.9
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
        while len(xs) < N:
            xs.append(w_init/2)
            ys.append(h_init/2)
            rs.append(0.1)
            
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        rs = np.array(rs[:N])
        
        x0_curr = np.concatenate(([w_init], xs, ys, rs))
        
        res = minimize(objective, x0_curr, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Extract best solution
    w_best = best_x[0]
    h_best = 2.0 - w_best
    xs_best = best_x[1:N+1]
    ys_best = best_x[N+1:2*N+1]
    rs_best = best_x[2*N+1:3*N+1]
    
    # Final polish: ensure no overlaps and valid bounds
    # If there are small overlaps due to tolerance, shrink radii slightly
    for i in range(N):
        for j in range(i+1, N):
            dx = xs_best[i] - xs_best[j]
            dy = ys_best[i] - ys_best[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = rs_best[i] + rs_best[j]
            if d < min_d:
                # Shrink both slightly
                factor = d / (min_d * 1.01) if d > 0 else 0.99
                rs_best[i] *= factor
                rs_best[j] *= factor
                
        # Ensure within bounds
        rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
        if rs_best[i] < 0:
            rs_best[i] = 0

    # Construct output array (x, y, r)
    result = np.column_stack((xs_best, ys_best, rs_best))
    
    # Calculate score for sanity check (sum of radii)
    # The evaluator calculates the score. We just return the array.
    
    return result
