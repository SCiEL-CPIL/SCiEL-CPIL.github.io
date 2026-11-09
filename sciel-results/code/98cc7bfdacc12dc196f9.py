import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 115.0 # 5s safety margin
    
    best_score = -np.inf
    best_x = None
    
    def objective(v):
        # v is [x1, y1, r1, x2, y2, r2, ...]
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Minimize negative sum of radii
        return -np.sum(r)
        
    def constraints_func(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        cons = []
        
        # Boundary constraints:
        # x - r >= 0
        # x + r <= 1  => 1 - x - r >= 0
        # y - r >= 0
        # y + r <= 1  => 1 - y - r >= 0
        # r >= eps
        cons.append(x - r)
        cons.append(1 - x - r)
        cons.append(y - r)
        cons.append(1 - y - r)
        cons.append(r - 1e-6)
        
        # Non-overlap constraints:
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                cons.append(dx*dx + dy*dy - dr*dr)
                
        return cons

    # We need to format constraints for SLSQP
    # SLSQP accepts a list of dicts: {'type': 'ineq', 'fun': lambda v: ...}
    # Or we can use a single function returning an array if we use a wrapper, 
    # but standard SLSQP interface is list of dicts.
    # Creating 429 dicts every time might be slow? No, it's fine.
    
    # To speed up, we can define the constraint functions once? 
    # The lambda closures capture i, j.
    
    # Alternative: Use a single vectorized constraint function?
    # scipy.optimize.minimize with SLSQP expects `constraints` as a sequence of dictionaries.
    
    # Let's build the constraint list
    constraints = []
    
    # Boundaries
    # We can add these as bounds in SLSQP if possible, but they depend on r.
    # x - r >= 0 is non-linear. So must be in constraints.
    
    # To reduce overhead, we can just use the vectorized approach if we switch to a method that supports it, 
    # but SLSQP is the most robust for this.
    
    # Let's try to optimize the constraint definition.
    # Instead of 429 separate lambda functions, we can define a function that returns the array of violations,
    # but SLSQP requires individual functions for `ineq` type in the list format.
    # However, there is a trick: we can use `method='SLSQP'` and pass a single constraint dict? No.
    
    # Actually, we can use `scipy.optimize.minimize` with `method='SLSQP'` and `constraints` as a list.
    # Let's generate the list.
    
    # To avoid regenerating the list of dicts for every restart (which is cheap), 
    # we do it once outside the loop? No, the functions depend on the indices, which are fixed.
    # So we can create the list of constraint dicts once.
    
    cons_list = []
    
    # Helper to add inequality constraint
    def add_incons(fun):
        cons_list.append({'type': 'ineq', 'fun': fun})
        
    # 1. Boundaries
    # x_i - r_i >= 0
    for i in range(N):
        add_incons(lambda v, i=i: v[3*i] - v[3*i+2])
        add_incons(lambda v, i=i: 1 - v[3*i] - v[3*i+2])
        add_incons(lambda v, i=i: v[3*i+1] - v[3*i+2])
        add_incons(lambda v, i=i: 1 - v[3*i+1] - v[3*i+2])
        add_incons(lambda v, i=i: v[3*i+2] - 1e-6)
        
    # 2. Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            idx_i = 3*i
            idx_j = 3*j
            def make_non_overlap(i, j):
                def fun(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dr = v[3*i+2] + v[3*j+2]
                    return dx*dx + dy*dy - dr*dr
                return fun
            add_incons(make_non_overlap(i, j))
            
    # Now cons_list is ready.
    
    # Initial guesses
    # 1. Random
    # 2. Grid
    # 3. Hexagonal
    
    initial_guesses = []
    
    # Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, side), np.linspace(0.1, 0.9, side))
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    if len(grid_x) >= N:
        initial_guesses.append(np.concatenate([grid_x[:N], grid_y[:N], np.full(N, 0.01)]))
        
    # Random
    for _ in range(5):
        rx = np.random.uniform(0.1, 0.9, N)
        ry = np.random.uniform(0.1, 0.9, N)
        rr = np.full(N, 0.01)
        initial_guesses.append(np.concatenate([rx, ry, rr]))
        
    # Run optimization for each guess
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, guess, method='SLSQP', 
                           bounds=[(0, 1)] * (2*N) + [(1e-6, 0.5)] * N,
                           constraints=cons_list,
                           options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_x = res.x
        except Exception as e:
            pass

    # If no valid result found, fallback to grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Reshape best_x to (N, 3)
    x = best_x[0::3]
    y = best_x[1::3]
    r = best_x[2::3]
    
    # Final validation and cleanup
    # Clip to ensure strictly inside
    r = np.clip(r, 1e-6, None)
    x = np.clip(x, r, 1-r)
    y = np.clip(y, r, 1-r)
    
    result = np.column_stack((x, y, r))
    return result
