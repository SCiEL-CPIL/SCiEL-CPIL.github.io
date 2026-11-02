import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Define initial configurations
    # Configuration 1: 5x5 grid minus 4 corners
    # Rows 0-4, Cols 0-4
    # Remove (0,0), (0,4), (4,0), (4,4)
    idxs = []
    for i in range(5):
        for j in range(5):
            if not (i in [0, 4] and j in [0, 4]):
                idxs.append((i, j))
    
    # We expect 21 indices
    assert len(idxs) == 21, f"Got {len(idxs)} indices"
    
    best_val = -np.inf
    best_res = None
    
    # Initial aspect ratios to try
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4]
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
            
        # w + h = 2
        # w / h = ar => w = ar * h => h (1 + ar) = 2 => h = 2 / (1 + ar)
        h_init = 2.0 / (1.0 + ar)
        w_init = ar * h_init
        
        # Initialize positions and radii
        # Grid spacing
        # The grid is 5x5, so 5 units wide, 5 units high in grid coords
        # Map grid coords to physical coords
        # x = (j + 0.5) * (w_init / 5.0)
        # y = (i + 0.5) * (h_init / 5.0)
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for k, (i, j) in enumerate(idxs):
            x0[k] = (j + 0.5) * (w_init / 5.0)
            y0[k] = (i + 0.5) * (h_init / 5.0)
            # Radius limited by cell size
            dx = w_init / 5.0
            dy = h_init / 5.0
            r0[k] = 0.5 * min(dx, dy) * 0.9 # 90% of min cell dimension
        
        # Variables: x, y, r, w, h
        # Total vars: 3*N + 2
        x_vars = np.concatenate([x0, y0, r0, [w_init, h_init]])
        
        def objective(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            w = v[3*N]
            h = v[3*N+1]
            return -np.sum(r)
            
        # Constraints
        # 1. w + h <= 2
        def c_perimeter(v):
            w = v[3*N]
            h = v[3*N+1]
            return 2.0 - (w + h)
            
        # 2. Bounds x, y, r
        # x >= r, x <= w - r
        # y >= r, y <= h - r
        # r >= 0
        
        # We can use bounds for r >= 0
        # But x, y constraints depend on w, h, r so they are nonlinear.
        
        def c_x_min(v):
            x = v[:N]
            r = v[2*N:3*N]
            return x - r # >= 0
            
        def c_x_max(v):
            x = v[:N]
            r = v[2*N:3*N]
            w = v[3*N]
            return w - r - x # >= 0
            
        def c_y_min(v):
            y = v[N:2*N]
            r = v[2*N:3*N]
            return y - r # >= 0
            
        def c_y_max(v):
            y = v[N:2*N]
            r = v[2*N:3*N]
            h = v[3*N+1]
            return h - r - y # >= 0
            
        # 3. Non-overlap
        def c_overlap(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            cons = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    r_sum = r[i] + r[j]
                    cons.append(dist_sq - r_sum*r_sum)
            return np.array(cons)
            
        constraints = [
            {'type': 'ineq', 'fun': c_perimeter},
            {'type': 'ineq', 'fun': c_x_min},
            {'type': 'ineq', 'fun': c_x_max},
            {'type': 'ineq', 'fun': c_y_min},
            {'type': 'ineq', 'fun': c_y_max},
            {'type': 'ineq', 'fun': c_overlap}
        ]
        
        # Bounds for variables
        # x, y can be anything, but constrained by others.
        # r >= 0
        # w, h > 0
        bounds = []
        for i in range(N):
            bounds.append((None, None)) # x
        for i in range(N):
            bounds.append((None, None)) # y
        for i in range(N):
            bounds.append((0, None)) # r
        bounds.append((0.1, 2.0)) # w
        bounds.append((0.1, 2.0)) # h
        
        try:
            res = minimize(
                objective, 
                x_vars, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if res.success:
                val = -res.fun
                # Check validity
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                w = res.x[3*N]
                h = res.x[3*N+1]
                
                # Verify perimeter
                if 2*(w+h) > 4.0 + 1e-6:
                    continue
                    
                # Verify bounds
                if np.any(x - r < -1e-6) or np.any(w - r - x < -1e-6):
                    continue
                if np.any(y - r < -1e-6) or np.any(h - r - y < -1e-6):
                    continue
                if np.any(r < -1e-6):
                    continue
                    
                # Verify overlap
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx*dx + dy*dy
                        r_sum = r[i] + r[j]
                        if dist_sq < r_sum*r_sum - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                            
                if valid and val > best_val:
                    best_val = val
                    best_res = res.x.copy()
                    
        except Exception as e:
            pass

    if best_res is None:
        # Fallback: return a very small valid configuration
        # 21 tiny circles in a 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract best result
    x = best_res[:N]
    y = best_res[N:2*N]
    r = best_res[2*N:3*N]
    w = best_res[3*N]
    h = best_res[3*N+1]
    
    # Ensure strict validity by shrinking radii slightly if needed
    # and centering if needed
    # Actually, if SLSQP succeeded with constraints, it should be valid.
    # But floating point errors can creep in.
    
    # Final check and slight shrink for safety
    min_margin = 1e-4
    # Check all constraints again
    # Perimeter
    if 2*(w+h) > 4.0:
        scale = 4.0 / (2*(w+h))
        x *= scale
        y *= scale
        r *= scale
        w *= scale
        h *= scale
        
    # Shrink radii by a tiny amount to ensure disjoint interiors
    r = r * (1 - 1e-5)
    
    # Construct output array (n, 3)
    out = np.column_stack((x, y, r))
    
    return out
