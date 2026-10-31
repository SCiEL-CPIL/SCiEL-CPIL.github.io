"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 26
    deadline = 150.0
    start = time.time()
    
    # Initial guess: grid of circles
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    
    # Convert to numpy array
    x0 = np.array(cells)
    
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Shape: (N, 3)
    
    def objective(v):
        # Maximize sum of radii -> minimize negative sum
        radii = v[2::3]
        return -np.sum(radii)
    
    def constraints(v):
        cons = []
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Boundary constraints: circle inside [0,1]^2
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})  # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})  # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})  # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})  # 1 - y - r >= 0
        
        # Non-overlap: (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                })
        
        # Positive radii
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
        
        return cons
    
    # Use SLSQP
    result = None
    best_val = -np.inf
    best_v = None
    
    # Multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            v0 = x0.flatten()
        else:
            # Perturb initial guess
            v0 = x0.flatten().copy()
            for k in range(N):
                v0[3*k] += np.random.uniform(-0.05, 0.05)
                v0[3*k+1] += np.random.uniform(-0.05, 0.05)
                v0[3*k+2] *= np.random.uniform(0.8, 1.2)
        
        try:
            res = minimize(objective, v0, method='SLSQP', constraints=constraints,
                          bounds=[(0, None) for _ in range(3*N)],
                          options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success or res.fun < best_val:
                # Check validity
                v = res.x
                xs = v[0::3]
                ys = v[1::3]
                rs = v[2::3]
                
                valid = True
                for i in range(N):
                    if xs[i] < -1e-6 or xs[i] > 1 + 1e-6 or ys[i] < -1e-6 or ys[i] > 1 + 1e-6:
                        valid = False
                        break
                    if rs[i] < 0:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_v = v.copy()
        except Exception as e:
            continue
    
    # Polish pass: fix positions, maximize radii
    if best_v is not None and time.time() < deadline - 10:
        try:
            def radius_objective(v):
                return -np.sum(v[2::3])
            
            def radius_constraints(v):
                cons = []
                xs = v[0::3]
                ys = v[1::3]
                rs = v[2::3]
                
                for i in range(N):
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
                
                for i in range(N):
                    for j in range(i+1, N):
                        cons.append({
                            'type': 'ineq',
                            'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                        })
                
                for i in range(N):
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
                
                return cons
            
            res2 = minimize(radius_objective, best_v, method='SLSQP', constraints=radius_constraints,
                          bounds=[(0, None) for _ in range(3*N)],
                          options={'maxiter': 500, 'ftol': 1e-12})
            
            if res2.fun < best_val:
                v = res2.x
                xs = v[0::3]
                ys = v[1::3]
                rs = v[2::3]
                
                valid = True
                for i in range(N):
                    if xs[i] < -1e-6 or xs[i] > 1 + 1e-6 or ys[i] < -1e-6 or ys[i] > 1 + 1e-6:
                        valid = False
                        break
                    if rs[i] < 0:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    best_v = v.copy()
        except:
            pass
    
    if best_v is None:
        # Fallback to initial grid
        return x0
    
    # Convert back to (N, 3) array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_v[3*i]
        result[i, 1] = best_v[3*i+1]
        result[i, 2] = best_v[3*i+2]
    
    # Final validity check and clamp
    for i in range(N):
        result[i, 0] = np.clip(result[i, 0], 0, 1)
        result[i, 1] = np.clip(result[i, 1], 0, 1)
        result[i, 2] = max(result[i, 2], 1e-6)
        # Ensure inside square
        result[i, 0] = np.clip(result[i, 0], result[i, 2], 1 - result[i, 2])
        result[i, 1] = np.clip(result[i, 1], result[i, 2], 1 - result[i, 2])
    
    return result
# EVOLVE-BLOCK-END
