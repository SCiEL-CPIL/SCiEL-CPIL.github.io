import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Decompose the 100-variable probability distribution P[y, x] into:
       - P_X[x]: marginal probability of x (10 vars)
       - P_Y_given_X[y, x]: conditional probability of y given x (10x10 vars, each row sums to 1)
       This reduces the effective search space and allows for more efficient gradient-free optimization.
       
    2. Use Differential Evolution (DE) on the reduced parameter space:
       - 9 free variables for P_X (10th is 1 - sum)
       - 90 free variables for P_Y_given_X (each of 10 rows has 9 free vars, 10th is 1 - sum)
       Total: 99 variables.
       
    3. Compute entropies efficiently using the decomposed structure.
       H(X) is determined solely by P_X.
       H(X-Y), H(X+Y), H(X+2Y) require summing over joint probabilities.
       
    4. Polish the best result with SLSQP on the full 100-variable space for final precision.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill

    # Precompute indices for efficient entropy calculation
    # We need to compute distributions of:
    # d = x - y
    # m0 = x
    # m1 = x + y
    # m2 = x + 2y
    
    # For a given P[y, x], the probability mass for a value v in the distribution of x + k*y is:
    # sum_{x, y: x + k*y = v} P[y, x]
    
    # To speed up DE evaluations, we can precompute the mapping from (x, y) to the bins for each distribution.
    
    d_val = np.zeros((n, n))
    m0_val = np.zeros((n, n))
    m1_val = np.zeros((n, n))
    m2_val = np.zeros((n, n))
    
    for x in range(n):
        for y in range(n):
            d_val[y, x] = x - y
            m0_val[y, x] = x
            m1_val[y, x] = x + y
            m2_val[y, x] = x + 2 * y

    d_vals, d_inv = np.unique(d_val.flatten(), return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val.flatten(), return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val.flatten(), return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val.flatten(), return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)
    
    # Flatten inverse mappings for bincount
    d_inv_flat = d_inv.flatten()
    m0_inv_flat = m0_inv.flatten()
    m1_inv_flat = m1_inv.flatten()
    m2_inv_flat = m2_inv.flatten()

    def shannon_entropy(p):
        p = p[p > 1e-15]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))

    def compute_score_from_p(p_flat):
        """
        Compute D/M for a flattened 100-dim probability vector.
        """
        p_d = np.bincount(d_inv_flat, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv_flat, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv_flat, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv_flat, weights=p_flat, minlength=num_m2)
        
        H_d = shannon_entropy(p_d)
        H_m0 = shannon_entropy(p_m0)
        H_m1 = shannon_entropy(p_m1)
        H_m2 = shannon_entropy(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        if M < 1e-9:
            return -1e9
        return H_d / M

    def objective_full(p_flat):
        return -compute_score_from_p(p_flat)

    # --- Phase 1: DE on reduced parameter space ---
    
    def decode_reduced(params):
        """
        Decode 99 parameters into a 100-dim probability vector.
        params[0:9] -> P_X (first 9, last is 1-sum)
        params[9:99] -> 10 rows of 9 params each for P_Y|X
        """
        p_x = np.zeros(n)
        p_x[:9] = params[0:9]
        p_x[9] = 1.0 - np.sum(p_x[:9])
        
        # Clip and normalize P_X just in case of numerical issues
        p_x = np.clip(p_x, 0, 1)
        sum_px = np.sum(p_x)
        if sum_px < 1e-9:
            p_x = np.ones(n) / n
        else:
            p_x /= sum_px
            
        p_full = np.zeros(100)
        
        for x in range(n):
            if p_x[x] < 1e-15:
                continue
            start_idx = 9 + x * 9
            p_y_cond = np.zeros(n)
            p_y_cond[:9] = params[start_idx:start_idx+9]
            p_y_cond[9] = 1.0 - np.sum(p_y_cond[:9])
            
            p_y_cond = np.clip(p_y_cond, 0, 1)
            sum_pyc = np.sum(p_y_cond)
            if sum_pyc < 1e-9:
                p_y_cond = np.ones(n) / n
            else:
                p_y_cond /= sum_pyc
                
            # P[y, x] = P_X[x] * P_Y|X[y, x]
            p_full[x*n : (x+1)*n] = p_x[x] * p_y_cond
            
        # Final normalization to ensure sum is exactly 1
        s = np.sum(p_full)
        if s < 1e-9:
            return np.ones(100) / 100.0
        return p_full / s

    def de_objective_reduced(params):
        p_full = decode_reduced(params)
        return objective_full(p_full)

    bounds_reduced = [(0.0, 1.0) for _ in range(99)]
    
    try:
        # DE on 99 variables
        # Use a smaller population and fewer iterations to fit in time, but rely on the reduced dimensionality
        res_de = differential_evolution(
            de_objective_reduced, 
            bounds_reduced, 
            seed=42,
            maxiter=150, 
            popsize=20,   
            tol=1e-6,
            polish=False, 
            init='sobol'  
        )
        
        p_best = decode_reduced(res_de.x)
        
    except Exception as e:
        p_best = np.ones(100) / 100.0

    if time.time() > deadline - 50.0:
        P = p_best.reshape((n, n))
        P = np.clip(P, 0.0, None)
        s = np.sum(P)
        if s > 1e-9:
            P = P / s
        else:
            P = np.ones((n, n)) / (n * n)
        return P

    # --- Phase 2: Local Refinement with SLSQP on full space ---
    
    starts = [p_best]
    starts.append(np.ones(100) / 100.0)
    
    # Add a structured start: "Anti-diagonal"
    p_adiag = np.zeros(100)
    for i in range(10):
        p_adiag[i*10+(9-i)] = 1.0/10.0
    starts.append(p_adiag)
    
    best_score = -1e9
    p_best_final = p_best
    
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    bounds_slsqp = [(0.0, 1.0) for _ in range(100)]
    
    for start_p in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                objective_full, 
                start_p, 
                method='SLSQP', 
                bounds=bounds_slsqp, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            
            p_res = res.x
            p_res = np.clip(p_res, 0.0, None)
            s = np.sum(p_res)
            if s < 1e-9: continue
            p_res /= s
            
            score = compute_score_from_p(p_res)
            
            if score > best_score:
                best_score = score
                p_best_final = p_res
                
        except Exception as e:
            pass

    # Final validation and normalization
    P = p_best_final.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P
