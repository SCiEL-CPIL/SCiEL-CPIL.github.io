import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Objective function to minimize: -sum(r)
    def objective(vars):
        w = vars[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 1e10
        # Check bounds for w
        if w < 0.1 or w > 1.9:
            return 1e10
            
        x = vars[1:1+N]
        y = vars[1+N:1+2*N]
        r = vars[1+2*N:1+3*N]
        
        # Sum of radii
        return -np.sum(r)

    # Constraints
    def constraints(vars):
        w = vars[0]
        h = 2.0 - w
        x = vars[1:1+N]
        y = vars[1+N:1+2*N]
        r = vars[1+2*N:1+3*N]
        
        cons = []
        
        # Circle i inside box
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+i] - v[1+2*N+i]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+i] - v[1+2*N+i]}) # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+N+i] - v[1+2*N+i]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[1+N+i] - v[1+2*N+i]}) # h - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+2*N+i]}) # r_i >= 0
            
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                             (v[1+i]-v[1+j])**2 + (v[1+N+i]-v[1+N+j])**2 - (v[1+2*N+i]+v[1+2*N+j])**2})
                             
        return cons

    # Initial guesses
    best_result = None
    best_val = -1e10
    
    # Guess 1: Grid 5x5 in 1x1 square
    side = 5
    w0 = 1.0
    x0, y0, r0 = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N: break
            x0.append((i + 0.5) / side)
            y0.append((j + 0.5) / side)
            r0.append(0.5 / side * 0.9) # Slightly smaller to avoid initial overlap
            count += 1
        if count >= N: break
        
    x0 = np.array(x0)
    y0 = np.array(y0)
    r0 = np.array(r0)
    
    # Scale to fit in 1x1
    # The grid above is in [0.1, 0.9]. 
    # Let's just use the values directly.
    
    vars0 = np.concatenate(([w0], x0, y0, r0))
    
    # Run optimization
    try:
        res = minimize(objective, vars0, method='SLSQP', constraints=constraints, 
                       bounds=[(0.5, 1.5)] + [(0.01, 1.0)]*3*N, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_result = res.x
    except Exception as e:
        pass
        
    # Guess 2: Hexagonal packing in 1x1 square
    # Generate hex points
    x_hex, y_hex, r_hex = [], [], []
    r_init = 0.1
    # Simple hex grid
    dx = 2 * r_init * 0.866
    dy = 2 * r_init
    for row in range(6):
        for col in range(6):
            x = col * dx + (row % 2) * dx / 2 + r_init
            y = row * dy + r_init
            if x < 1.0 and y < 1.0:
                x_hex.append(x)
                y_hex.append(y)
                r_hex.append(r_init)
            if len(x_hex) >= N:
                break
        if len(x_hex) >= N:
            break
            
    if len(x_hex) >= N:
        x_hex = np.array(x_hex[:N])
        y_hex = np.array(y_hex[:N])
        r_hex = np.array(r_hex[:N])
        vars0_hex = np.concatenate(([1.0], x_hex, y_hex, r_hex))
        try:
            res = minimize(objective, vars0_hex, method='SLSQP', constraints=constraints, 
                           bounds=[(0.5, 1.5)] + [(0.01, 1.0)]*3*N, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_result = res.x
        except Exception as e:
            pass

    if best_result is not None:
        w = best_result[0]
        x = best_result[1:1+N]
        y = best_result[1+N:1+2*N]
        r = best_result[1+2*N:1+3*N]
        
        # Final check and cleanup
        # Ensure validity
        valid = True
        for i in range(N):
            if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > (2-w) - r[i] or r[i] <= 0:
                valid = False
                break
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                sum_r = r[i] + r[j]
                if dist2 < sum_r**2 - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            return np.column_stack((x, y, r))
        else:
            # Fallback to seed if optimization failed or produced invalid result
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
