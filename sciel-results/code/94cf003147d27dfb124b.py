import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Guess: Hexagonal Packing
    row_lengths = [6, 5, 6, 5, 6, 4]
    r_init = 0.09
    
    centers = []
    radii = []
    
    y_pos = r_init
    for i, length in enumerate(row_lengths):
        start_x = (1 - (length - 1) * 2 * r_init) / 2
        for j in range(length):
            x = start_x + j * 2 * r_init
            centers.append([x, y_pos])
            radii.append(r_init)
        y_pos += r_init * np.sqrt(3)
        
    x0 = np.array(centers + radii).flatten()
    
    def objective(params):
        radii_vec = params[2::3]
        return -np.sum(radii_vec)
        
    def boundary_constraints(params):
        cons = np.zeros(4 * N)
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            cons[4*i] = x - r
            cons[4*i + 1] = y - r
            cons[4*i + 2] = 1 - x - r
            cons[4*i + 3] = 1 - y - r
        return cons
        
    def overlap_constraints(params):
        # N=32, pairs = 32*31/2 = 496
        cons = np.zeros(N * (N - 1) // 2)
        idx = 0
        for i in range(N):
            xi = params[3*i]
            yi = params[3*i + 1]
            ri = params[3*i + 2]
            for j in range(i + 1, N):
                xj = params[3*j]
                yj = params[3*j + 1]
                rj = params[3*j + 2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                cons[idx] = dist_sq - sum_r**2
                idx += 1
        return cons
        
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    bounds = [(0, 1) for _ in range(2*N)] + [(0, 0.5) for _ in range(N)]
    
    best_result = None
    best_score = -np.inf
    
    # Run optimization
    # Use a single run with a good start first. If time permits, try perturbations.
    
    # Run 1: Original start
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-10})
    
    if res.success:
        best_score = -res.fun
        best_result = res.x
        
    # Run 2: Perturbed start if time allows
    if time.time() < deadline - 20:
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            noise = np.random.normal(0, 0.02, x0.shape)
            x_start = x0 + noise
            x_start[:2*N] = np.clip(x_start[:2*N], 0.01, 0.99)
            x_start[2::3] = np.clip(x_start[2::3], 0.01, 0.4)
            
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_result = res.x

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    out = np.zeros((N, 3))
    for i in range(N):
        out[i, 0] = best_result[3*i]
        out[i, 1] = best_result[3*i + 1]
        out[i, 2] = best_result[3*i + 2]
        
    return out
