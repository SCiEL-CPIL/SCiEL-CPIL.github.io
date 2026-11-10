import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Regular octagon vertices on unit circle
    # P0 at (-1, 0), P4 at (1, 0)
    angles = [np.pi + 2 * np.pi * k / 8 for k in range(8)]
    P_reg = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Fixed points
    P0 = P_reg[0]
    P4 = P_reg[4]
    
    # Initial guess for the other 6 points
    P_init = P_reg.copy()
    
    def get_points(flat):
        # flat is 12 floats
        P = P_init.copy()
        P[1] = flat[0:2]
        P[2] = flat[2:4]
        P[3] = flat[4:6]
        P[5] = flat[6:8]
        P[6] = flat[8:10]
        P[7] = flat[10:12]
        return P
    
    def objective(flat):
        P = get_points(flat)
        # Calculate distances
        n = 8
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(P[i] - P[j])
                dists.append(d)
        dists = np.array(dists)
        
        # Handle zero distances
        if np.any(dists < 1e-10):
            return 1e10
        
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
            
        # Log product
        log_prod = np.sum(np.log(dists))
        
        # Score to maximize: log_prod - 28 * log(D)
        # We minimize the negative
        return -(log_prod - 28 * np.log(D))
        
    best_score = -np.inf
    best_flat = P_init[1].flatten() # Placeholder, need to extract correctly
    
    # Extract initial flat vector
    def to_flat(P):
        return np.concatenate([P[1], P[2], P[3], P[5], P[6], P[7]])
    
    x0 = to_flat(P_init)
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_flat = res.x
                
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline - 10:
        # Return best found or regular octagon
        if best_flat is not None and best_score > -np.inf:
             P_final = get_points(best_flat)
        else:
             P_final = P_reg
        return P_final.tolist()
        
    # Try a few random restarts if time permits
    for i in range(5):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial guess
        noise = np.random.normal(0, 0.1, size=12)
        x0_pert = x0 + noise
        
        try:
            res = minimize(objective, x0_pert, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_flat = res.x
                    
        except Exception:
            pass
            
    # Finalize
    if best_flat is not None and best_score > -np.inf:
        P_final = get_points(best_flat)
    else:
        P_final = P_reg
        
    # Ensure distinctness
    # If points are too close, nudge them
    # But regular octagon is distinct. Optimized should be too.
    
    return P_final.tolist()
