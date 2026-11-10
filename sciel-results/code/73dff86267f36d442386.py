# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    # Variables: x1, y1, s1, x2, y2, s2, ..., x10, y10, s10
    # Total 3*n variables
    
    # Smooth penalty for overlap
    # A = (si+sj)/2 - |xi-xj|
    # B = (si+sj)/2 - |yi-yj|
    # Overlap if A > 0 and B > 0
    # Penalty = softplus(A) * softplus(B)
    
    def softplus(x, k=10.0):
        # Stable softplus
        return np.where(x > 0, x + np.log1p(np.exp(-k*x))/k, np.log1p(np.exp(k*x))/k)
        # Wait, standard softplus is log(1+exp(x)). 
        # To make it sharper, we scale x by k.
        # sp(x) = (1/k) * log(1 + exp(k*x))
        # Let's use a simpler version that is stable.
        
    def objective(vars):
        # vars: [x1, y1, s1, x2, y2, s2, ...]
        x = vars[0::3]
        y = vars[1::3]
        s = vars[2::3]
        
        # Objective: -sum(s) + M * sum(overlap_penalties)
        score = -np.sum(s)
        
        # Overlap penalties
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # A_ij
                A = (s[i] + s[j]) / 2.0 - np.abs(x[i] - x[j])
                # B_ij
                B = (s[i] + s[j]) / 2.0 - np.abs(y[i] - y[j])
                
                # Smooth max(0, A)
                # Using softplus with sharpness k
                k = 20.0
                # sp(A) approx max(0, A)
                # For stability:
                # if A > 0: A + log(1+exp(-kA))/k
                # if A <= 0: log(1+exp(kA))/k
                sp_A = np.where(A > 0, A + np.log1p(np.exp(-k*A))/k, np.log1p(np.exp(k*A))/k)
                sp_B = np.where(B > 0, B + np.log1p(np.exp(-k*B))/k, np.log1p(np.exp(k*B))/k)
                
                penalty += sp_A * sp_B
        
        M = 1000.0
        return score + M * penalty

    # Bounds
    # x, y in [0, 1]
    # s in [0, 1]
    # Actually, to stay inside [0,1], we need:
    # s/2 <= x <= 1 - s/2
    # This is a non-linear constraint on bounds.
    # L-BFGS-B requires simple bounds.
    # We can set x, y in [0, 1] and s in [0, 1].
    # And add boundary violation penalty to the objective.
    
    def objective_with_boundary(vars):
        x = vars[0::3]
        y = vars[1::3]
        s = vars[2::3]
        
        score = -np.sum(s)
        
        # Boundary penalty
        # x - s/2 >= 0 => s/2 - x <= 0
        # 1 - x - s/2 >= 0 => x + s/2 - 1 <= 0
        # Violation if x < s/2 or x > 1 - s/2
        # dist_left = s/2 - x
        # dist_right = x + s/2 - 1
        # penalty = max(0, dist_left)^2 + max(0, dist_right)^2
        
        k = 20.0
        def sp(x):
            return np.where(x > 0, x + np.log1p(np.exp(-k*x))/k, np.log1p(np.exp(k*x))/k)
            
        pen_bound = 0.0
        for i in range(n):
            dl = s[i]/2.0 - x[i]
            dr = x[i] + s[i]/2.0 - 1.0
            dt = s[i]/2.0 - y[i]
            db = y[i] + s[i]/2.0 - 1.0
            pen_bound += sp(dl)**2 + sp(dr)**2 + sp(dt)**2 + sp(db)**2
            
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                A = (s[i] + s[j]) / 2.0 - np.abs(x[i] - x[j])
                B = (s[i] + s[j]) / 2.0 - np.abs(y[i] - y[j])
                sp_A = sp(A)
                sp_B = sp(B)
                pen_overlap += sp_A * sp_B
                
        M_bound = 1000.0
        M_overlap = 1000.0
        
        return score + M_bound * pen_bound + M_overlap * pen_overlap

    bounds = [(0, 1)] * (3 * n)
    
    # Initial guesses
    # 1. Grid
    k_grid = 3
    side_grid = 0.9 / k_grid # Slightly smaller to allow growth
    idx = np.arange(n)
    cols = idx % k_grid
    rows = idx // k_grid
    cx_init = (cols + 0.5) * (1.0/k_grid)
    cy_init = (rows + 0.5) * (1.0/k_grid)
    
    x_init = cx_init.copy()
    y_init = cy_init.copy()
    s_init = np.full(n, side_grid)
    
    vars_init = np.empty(3*n)
    vars_init[0::3] = x_init
    vars_init[1::3] = y_init
    vars_init[2::3] = s_init
    
    best_vars = vars_init
    best_score = -np.inf
    
    start_time = time.time()
    time_limit = 500.0 # Leave 100s for overhead/safety
    
    # Run multiple restarts
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Add noise for restarts > 0
        if restart == 0:
            v0 = vars_init
        else:
            v0 = vars_init + np.random.normal(0, 0.05, 3*n)
            # Clip to bounds
            v0 = np.clip(v0, 0, 1)
            
        res = minimize(objective_with_boundary, v0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        # Check validity regardless of res.success, as L-BFGS-B might stop early
        v_res = res.x
        s_res = v_res[2::3]
        
        # Calculate true score (sum of sides)
        current_sum_s = np.sum(s_res)
        
        # Verify constraints strictly to ensure validity
        x_r = v_res[0::3]
        y_r = v_res[1::3]
        valid = True
        tol = 1e-7
        
        # Boundary check: must lie inside [-tol, 1+tol]
        for i in range(n):
            if s_res[i] <= 0:
                valid = False
                break
            if x_r[i] - s_res[i]/2 < -tol or x_r[i] + s_res[i]/2 > 1 + tol:
                valid = False
                break
            if y_r[i] - s_res[i]/2 < -tol or y_r[i] + s_res[i]/2 > 1 + tol:
                valid = False
                break
        
        if valid:
            # Check overlaps: forbidden if |dx| < t and |dy| < t where t = (si+sj)/2 - tol
            for i in range(n):
                for j in range(i+1, n):
                    t = (s_res[i] + s_res[j])/2.0 - tol
                    # If t is negative, they can't overlap in the forbidden sense (touching or separated is fine, 
                    # actually if t<0, the condition |dx| < t is impossible for positive |dx| unless t>0? 
                    # Wait, if s_i+s_j is very small, t can be negative.
                    # The condition for overlap is |dx| < (s_i+s_j)/2 AND |dy| < (s_i+s_j)/2.
                    # The prompt says overlap is forbidden if |dx| < t and |dy| < t with t = (s_i+s_j)/2 - tol.
                    # This is a slightly relaxed definition of overlap to allow numerical tolerance.
                    if abs(x_r[i] - x_r[j]) < t and abs(y_r[i] - y_r[j]) < t:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and current_sum_s > best_score:
                best_score = current_sum_s
                best_vars = v_res
    
    # Extract best result
    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    s_best = best_vars[2::3]
    
    centers = np.stack([x_best, y_best], axis=1)
    sides = s_best
    
    return centers, sides

# EVOLVE-BLOCK-END
