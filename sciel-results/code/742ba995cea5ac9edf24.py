import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 21
    
    # Variables: x_1..x_n, y_1..y_n, r_1..r_n
    # Objective: Minimize -sum(r_i)
    # Constraints:
    # 1. x_i >= r_i
    # 2. x_i <= 1 - r_i
    # 3. y_i >= r_i
    # 4. y_i <= 1 - r_i
    # 5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i < j
    
    # Initial guess: Hexagonal packing in a square
    rows = [5, 4, 5, 4, 3]
    centers = []
    d = 0.1
    h = np.sqrt(3) / 2 * d
    current_y = d / 2
    for i, count in enumerate(rows):
        start_x = (1.0 - (count - 1) * d) / 2.0
        for j in range(count):
            x = start_x + j * d
            y = current_y
            centers.append((x, y))
        current_y += h
        
    centers = np.array(centers)
    r_init = np.full(n, d / 2)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = r_init[i]
        
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
        
    def objective_grad(v):
        # Gradient of -sum(r)
        # d(-sum(r))/dr_i = -1
        # d(-sum(r))/dx_i = 0
        # d(-sum(r))/dy_i = 0
        g = np.zeros(3 * n)
        g[2::3] = -1.0
        return g
        
    # Constraints
    # We use inequality constraints: fun(x) >= 0
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # Overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def constraints(v):
        c = []
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Boundary constraints
        c.append(x - r)
        c.append(1.0 - x - r)
        c.append(y - r)
        c.append(1.0 - y - r)
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_r_sq = (r[i] + r[j])**2
                c.append(dist_sq - sum_r_sq)
                
        return np.concatenate(c)
        
    def constraints_jac(v):
        # Jacobian of constraints
        # Rows:
        # 0..n-1: x_i - r_i
        # n..2n-1: 1 - x_i - r_i
        # 2n..3n-1: y_i - r_i
        # 3n..4n-1: 1 - y_i - r_i
        # 4n..4n + m - 1: Overlaps
        
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        jac = np.zeros((4*n + n*(n-1)//2, 3*n))
        
        # x_i - r_i
        # d/dx_i = 1, d/dr_i = -1
        for i in range(n):
            jac[i, 3*i] = 1.0
            jac[i, 3*i+2] = -1.0
            
        # 1 - x_i - r_i
        # d/dx_i = -1, d/dr_i = -1
        for i in range(n):
            jac[n+i, 3*i] = -1.0
            jac[n+i, 3*i+2] = -1.0
            
        # y_i - r_i
        # d/dy_i = 1, d/dr_i = -1
        for i in range(n):
            jac[2*n+i, 3*i+1] = 1.0
            jac[2*n+i, 3*i+2] = -1.0
            
        # 1 - y_i - r_i
        # d/dy_i = -1, d/dr_i = -1
        for i in range(n):
            jac[3*n+i, 3*i+1] = -1.0
            jac[3*n+i, 3*i+2] = -1.0
            
        # Overlaps
        row = 4*n
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                sr = r[i] + r[j]
                
                # fun = dx^2 + dy^2 - sr^2
                # d/dx_i = 2*dx, d/dx_j = -2*dx
                # d/dy_i = 2*dy, d/dy_j = -2*dy
                # d/dr_i = -2*sr, d/dr_j = -2*sr
                
                jac[row, 3*i] = 2*dx
                jac[row, 3*j] = -2*dx
                jac[row, 3*i+1] = 2*dy
                jac[row, 3*j+1] = -2*dy
                jac[row, 3*i+2] = -2*sr
                jac[row, 3*j+2] = -2*sr
                row += 1
                
        return jac
        
    # Bounds:
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Run optimization
    best_result = None
    best_val = -np.inf
    
    # Try a few restarts or just one long run?
    # Trust-constr can be slow. Let's try one run with high maxiter.
    # If it fails, return the initial valid config if any, or a small config.
    
    # Ensure initial point is feasible or close to it
    # The hexagonal packing with d=0.1 should be feasible in [0,1]x[0,1]
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='trust-constr', 
            jac=objective_grad, 
            hess=None, # Let it approximate
            constraints=[{'type': 'ineq', 'fun': constraints, 'jac': constraints_jac}],
            bounds=bounds,
            options={'maxiter': 200, 'gtol': 1e-6, 'xtol': 1e-6}
        )
        
        if res.success or (res.fun < best_val):
            # Check validity of result
            v = res.x
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            # Check if valid
            valid = True
            if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6) or \
               np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
                valid = False
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                        if dist_sq < (r[i] + r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = res.x
                    
    except Exception:
        pass
        
    # If optimization failed or didn't find a valid better solution, use fallback
    if best_result is None:
        # Try to find a valid configuration from initial guess or shrink it
        x = centers[:, 0]
        y = centers[:, 1]
        r = r_init.copy()
        
        # Shrink r until valid
        for _ in range(100):
            valid = True
            if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6) or \
               np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
                valid = False
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                        if dist_sq < (r[i] + r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                v = np.zeros(3*n)
                v[0::3] = x
                v[1::3] = y
                v[2::3] = r
                best_result = v
                best_val = np.sum(r)
                break
            else:
                r *= 0.95
                if np.sum(r) < 0.01:
                    # Give up on this config, use tiny circles
                    r = np.full(n, 0.01)
                    v = np.zeros(3*n)
                    v[0::3] = x
                    v[1::3] = y
                    v[2::3] = r
                    best_result = v
                    best_val = np.sum(r)
                    break
        else:
            # Fallback: tiny circles
            r = np.full(n, 0.01)
            v = np.zeros(3*n)
            v[0::3] = x
            v[1::3] = y
            v[2::3] = r
            best_result = v
            best_val = np.sum(r)
            
    # Format output
    x_out = best_result[0::3]
    y_out = best_result[1::3]
    r_out = best_result[2::3]
    
    result = np.column_stack((x_out, y_out, r_out))
    return result
