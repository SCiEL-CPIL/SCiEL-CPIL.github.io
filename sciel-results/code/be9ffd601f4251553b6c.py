import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # Leave 3s for safety margin

    n = 8
    # Precompute indices for unordered pairs
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]

    # Constants for the objective function
    # log(Score) = 56*log(2) + 2*sum(log(d_ij)) - 56*log(D) - 8*log(8)
    # We minimize -log(Score)
    LOG_2 = np.log(2.0)
    LOG_8 = np.log(8.0)
    CONST_OFFSET = 56.0 * LOG_2 - 8.0 * LOG_8

    def objective_5d(params):
        if time.time() > deadline:
            return 1e10
        r1, r2, theta1, theta2, dy = params
        
        # Ensure radii are positive
        r1 = abs(r1)
        r2 = abs(r2)
        
        # Generate points for Square 1
        angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Generate points for Square 2
        angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2) + dy])
        
        points = np.vstack([p1, p2])
        
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        if np.any(dists < 1e-12):
            return 1e10
            
        D = np.max(dists)
        if D < 1e-12:
            return 1e10
            
        sum_log_d = np.sum(np.log(dists))
        log_score = CONST_OFFSET + 2.0 * sum_log_d - 56.0 * np.log(D)
        return -log_score

    # Bounds for the 5 parameters
    # Radii: 0 to 5
    # Angles: 0 to pi/2 (due to symmetry of squares)
    # dy: -2 to 2
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, np.pi/2), (0.0, np.pi/2), (-2.0, 2.0)]
    
    # Initial guess: Regular Octagon corresponds to r1=r2, theta2=theta1+pi/4, dy=0
    # r = 1 for unit circumradius (diameter 2)
    x0 = [1.0, 1.0, 0.0, np.pi/4, 0.0]
    
    best_val = float('inf')
    best_params = x0
    
    # Phase 1: Differential Evolution on the 5D manifold
    if time.time() < deadline - 20.0:
        try:
            res = differential_evolution(
                objective_5d,
                bounds,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                seed=42,
                init='sobol',
                polish=False
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Phase 2: Local Refinement
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                objective_5d,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Phase 3: Final Polish with Powell
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                objective_5d,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Reconstruct points
    r1, r2, theta1, theta2, dy = best_params
    r1 = abs(r1)
    r2 = abs(r2)
    
    angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2) + dy])
    
    points = np.vstack([p1, p2])
    
    return [[float(p[0]), float(p[1])] for p in points]
