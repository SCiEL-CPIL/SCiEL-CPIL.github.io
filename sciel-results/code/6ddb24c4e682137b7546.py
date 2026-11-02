import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Variables: x[0:N], y[0:N], r[0:N], t
    # Total 3*N + 1 variables
    n_vars = 3 * N + 1
    
    def get_dims(t):
        t = np.clip(t, 0.1, 0.9)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def objective(vars):
        x = vars[0:N]
        y = vars[N:2*N]
        r = vars[2*N:3*N]
        t = vars[3*N]
        
        w, h = get_dims(t)
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        score = np.sum(r)
        
        # Penalty for boundary violations
        penalty = 0.0
        # x bounds
        penalty += np.sum(np.maximum(0, r - x)**2)
        penalty += np.sum(np.maximum(0, r - (w - x))**2)
        # y bounds
        penalty += np.sum(np.maximum(0, r - y)**2)
        penalty += np.sum(np.maximum(0, r - (h - y))**2)
        
        # Penalty for circle-circle overlaps
        # Vectorized calculation
        # x_diff = x[:, None] - x[None, :]
        # y_diff = y[:, None] - y[None, :]
        # dist = sqrt(x_diff**2 + y_diff**2)
        # sum_r = r[:, None] + r[None, :]
        # overlaps = max(0, sum_r - dist)
        # penalty += sum(overlaps**2)
        
        # To save time, do it in a loop or vectorized carefully
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        # Avoid sqrt of 0
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        sum_r = r[:, None] + r[None, :]
        # Only upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlaps = np.maximum(0, sum_r[mask] - dist[mask])
        penalty += np.sum(overlaps**2)
        
        # We want to maximize score, so minimize -score + penalty
        # The penalty needs to be strong enough
        return -score + 10.0 * penalty

    def is_valid(x, y, r, t):
        w, h = get_dims(t)
        if np.any(r < 1e-6):
            return False
        if np.any(x < r) or np.any(x > w - r):
            return False
        if np.any(y < r) or np.any(y > h - r):
            return False
        
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        dist = np.sqrt(dist_sq)
        sum_r = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any(dist[mask] < sum_r[mask] - 1e-9):
            return False
        return True

    def get_value(x, y, r, t):
        return np.sum(r)

    best_val = 0.0
    best_vars = None
    
    # Initial guesses
    # 1. Square grid 5x5
    side = 5
    x0 = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)])
    y0 = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)])
    r0 = np.full(N, 0.5 / side * 0.9)
    t0 = 0.5
    vars0 = np.concatenate([x0, y0, r0, [t0]])
    
    # 2. Rectangle 4x5 grid + 1
    # 4 cols, 5 rows. w=1.2, h=0.8? 
    # Let's try t=0.6 (w=1.2, h=0.8)
    # 5 rows, 4 cols.
    # x: 4 cols, y: 5 rows.
    # 20 circles in grid, 1 in center?
    # This is complex to set up. Let's stick to square grid and maybe a few random perturbations.
    
    # Try a few random initializations around the grid
    rng = np.random.default_rng(42)
    
    candidates = []
    candidates.append(vars0)
    
    # Add some random perturbations
    for _ in range(5):
        x_pert = x0 + rng.uniform(-0.1, 0.1, N)
        y_pert = y0 + rng.uniform(-0.1, 0.1, N)
        r_pert = r0 + rng.uniform(-0.02, 0.02, N)
        t_pert = 0.5 + rng.uniform(-0.1, 0.1)
        vars_pert = np.concatenate([x_pert, y_pert, r_pert, [t_pert]])
        candidates.append(vars_pert)
        
    # Also try a more rectangular aspect
    # 5x4 grid in a 1.2 x 0.8 box?
    # w=1.2, h=0.8. t=0.6.
    # 4 cols, 5 rows.
    x_rect = np.array([(i + 0.5) / 4 * 1.2 for i in range(4) for _ in range(5)])
    y_rect = np.array([(j + 0.5) / 5 * 0.8 for _ in range(4) for j in range(5)])
    r_rect = np.full(N, 0.5/4 * 0.9) # limited by width
    # 21st circle? Maybe in a gap?
    # Just use the first 21 from a 5x5 grid stretched?
    # Let's just use the square grid candidates for now, Nelder-Mead can adjust t.
    
    for v0 in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, v0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.success:
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                t = res.x[3*N]
                
                if is_valid(x, y, r, t):
                    val = get_value(x, y, r, t)
                    if val > best_val:
                        best_val = val
                        best_vars = res.x
        except Exception:
            pass

    if best_vars is None:
        # Fallback to seed-like solution
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x = best_vars[0:N]
    y = best_vars[N:2*N]
    r = best_vars[2*N:3*N]
    t = best_vars[3*N]
    
    # Final safety check and adjustment
    # If slightly invalid due to floating point, shrink radii
    w, h = get_dims(t)
    # Check constraints
    min_margin = np.min(np.minimum(np.minimum(x, w-x), np.minimum(y, h-y)))
    if min_margin < 1e-6:
         # Scale down
         scale = 0.99
         x = x * scale + (w*(1-scale))/2 # Keep centered? No, just scale radii
         # Better: reduce radii
         r = r * 0.99
    
    # Re-check validity
    if not is_valid(x, y, r, t):
        # Aggressive shrink
        r = r * 0.9
        if not is_valid(x, y, r, t):
            # Fallback
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    result = np.column_stack((x, y, r))
    return result

