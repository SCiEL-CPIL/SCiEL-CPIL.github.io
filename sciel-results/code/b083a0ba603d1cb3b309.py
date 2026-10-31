import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer
    
    best_score = -np.inf
    best_result = None
    
    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def objective(z):
        x, y, r = unpack(z)
        return -np.sum(r)

    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[3*i] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[3*i+1] - z[3*i+2]})
        
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i+1, N):
            def dist_constraint(z, i=i, j=j):
                xi, xj = z[3*i], z[3*j]
                yi, yj = z[3*i+1], z[3*j+1]
                ri, rj = z[3*i+2], z[3*j+2]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            constraints.append({'type': 'ineq', 'fun': dist_constraint})

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    n_starts = 15
    np.random.seed(42)
    
    for k in range(n_starts):
        if time.time() > deadline:
            break
            
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            x0[i] = (col + 0.5) / side + np.random.uniform(-0.2, 0.2)
            y0[i] = (row + 0.5) / side + np.random.uniform(-0.2, 0.2)
            r0[i] = 0.15
            
        x0 = np.clip(x0, 0.1, 0.9)
        y0 = np.clip(y0, 0.1, 0.9)
        
        z0 = np.zeros(3*N)
        z0[0::3] = x0
        z0[1::3] = y0
        z0[2::3] = r0
        
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 150, 'ftol': 1e-8})
            
            if res.success:
                x, y, r = unpack(res.x)
                feasible = True
                for i in range(N):
                    if r[i] < -1e-5 or x[i] - r[i] < -1e-5 or x[i] + r[i] > 1 + 1e-5 or y[i] - r[i] < -1e-5 or y[i] + r[i] > 1 + 1e-5:
                        feasible = False
                        break
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-5:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = res.x.copy()
                        
        except Exception:
            pass

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_result[0::3]
    y = best_result[1::3]
    r = best_result[2::3]
    
    r = np.clip(r, 0, 0.5)
    x = np.clip(x, r, 1-r)
    y = np.clip(y, r, 1-r)
    
    result = np.column_stack((x, y, r))
    return result
