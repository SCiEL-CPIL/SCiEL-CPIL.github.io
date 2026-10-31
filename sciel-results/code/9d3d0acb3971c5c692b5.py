import numpy as np
from scipy.optimize import minimize

N = 26

def _objective(v):
    """Minimize negative sum of radii."""
    # v is [x1, y1, r1, ..., xN, yN, rN]
    radii = v[2::3]
    return -np.sum(radii)

def _constraints(v):
    """Return list of constraint functions (inequality >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    cons = []
    # Wall constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda i=i, x=x, r=r: x[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i, x=x, r=r: 1 - x[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i, y=y, r=r: y[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i, y=y, r=r: 1 - y[i] - r[i]})
        
    # Pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            def fun(i=i, j=j, x=x, y=y, r=r):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                return dx*dx + dy*dy - (r[i] + r[j])**2
            cons.append({'type': 'ineq', 'fun': fun})
            
    return cons

def _bounds():
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    return bounds

def construct():
    # Multiple restarts to find global optimum
    best_result = None
    best_val = -np.inf
    
    # Initial guesses
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    x_grid, y_grid = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), 
                                  np.linspace(0.5/side, 1-0.5/side, side))
    x_flat = x_grid.flatten()[:N]
    y_flat = y_grid.flatten()[:N]
    r_init = np.full(N, 0.05)
    v0_grid = np.concatenate([x_flat, y_flat, r_init])
    
    # 2. Random
    np.random.seed(42)
    v0_rand = np.concatenate([np.random.rand(N), np.random.rand(N), np.full(N, 0.05)])
    
    # 3. Hex-like
    # Just use random for now, grid is usually good enough for SLSQP
    
    starts = [v0_grid, v0_rand]
    
    for v0 in starts:
        try:
            res = minimize(_objective, v0, method='SLSQP', bounds=_bounds(), 
                           constraints=_constraints(v0), options={'maxiter': 500, 'ftol': 1e-9})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_result = res.x
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to grid
        return np.column_stack([x_flat, y_flat, r_init])
        
    x = best_result[0::3]
    y = best_result[1::3]
    r = best_result[2::3]
    
    return np.column_stack((x, y, r))
