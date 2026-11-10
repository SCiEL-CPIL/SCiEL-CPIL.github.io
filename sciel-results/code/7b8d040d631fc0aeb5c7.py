import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 axis-aligned squares in the unit square.
    Optimizes in center coordinates with penalty-based approach.
    """
    n = 10
    
    def objective(z):
        cx = z[0::3]
        cy = z[1::3]
        s = z[2::3]
        
        obj = -np.sum(s)
        penalty = 0.0
        
        # Boundary penalties: square must be within [0,1]^2
        # cx - s/2 >= 0, cx + s/2 <= 1, same for cy
        v_left = np.maximum(0.0, s/2.0 - cx)
        v_right = np.maximum(0.0, cx - (1.0 - s/2.0))
        v_bottom = np.maximum(0.0, s/2.0 - cy)
        v_top = np.maximum(0.0, cy - (1.0 - s/2.0))
        
        penalty += np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Overlap penalties: penalize when squares overlap in both x and y
        for i in range(n):
            for j in range(i+1, n):
                req = (s[i] + s[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                ox = max(0.0, req - dx)
                oy = max(0.0, req - dy)
                # Product is zero unless both are positive (i.e., overlap in both dims)
                penalty += ox * oy
        
        obj += 5000.0 * penalty
        return obj
    
    def make_grid_init(k=4):
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        z = np.zeros(n*3)
        z[0::3] = cx
        z[1::3] = cy
        z[2::3] = s
        return z
    
    def make_random_init():
        cx = np.random.uniform(0.15, 0.85, n)
        cy = np.random.uniform(0.15, 0.85, n)
        s = np.random.uniform(0.18, 0.38, n)
        z = np.zeros(n*3)
        z[0::3] = cx
        z[1::3] = cy
        z[2::3] = s
        return z
    
    bounds = [(0.0, 1.0)] * (n*3)
    
    best_z = None
    best_score = -np.inf
    
    starts = [make_grid_init(4), make_grid_init(3), make_random_init(), make_random_init(), make_random_init()]
    
    for z0 in starts:
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            z = res.x
            cx = z[0::3]
            cy = z[1::3]
            s = z[2::3]
            
            valid = True
            tol = 1e-7
            for i in range(n):
                if not np.isfinite(s[i]) or s[i] <= 0:
                    valid = False; break
                if cx[i] - s[i]/2.0 < -tol or cx[i] + s[i]/2.0 > 1.0 + tol:
                    valid = False; break
                if cy[i] - s[i]/2.0 < -tol or cy[i] + s[i]/2.0 > 1.0 + tol:
                    valid = False; break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        t = (s[i] + s[j]) / 2.0 - tol
                        if abs(cx[i]-cx[j]) < t and abs(cy[i]-cy[j]) < t:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(s)
                if score > best_score:
                    best_score = score
                    best_z = z.copy()
        except Exception:
            pass
    
    if best_z is not None:
        cx = best_z[0::3]
        cy = best_z[1::3]
        s = best_z[2::3]
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = s.astype(np.float64)
        return centers, sides
    else:
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
