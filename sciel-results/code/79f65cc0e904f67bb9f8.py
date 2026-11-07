import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def _calc_max_corr(h):
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _local_search_binary(h_bin):
    h_bin = np.array(h_bin, dtype=np.int8)
    current_max = _calc_max_corr(h_bin.astype(np.float64))
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_i = -1
        best_j = -1
        best_new_h = None
        
        for i in ones:
            for j in zeros:
                h_temp = h_bin.copy()
                h_temp[i] = 0
                h_temp[j] = 1
                
                new_max = _calc_max_corr(h_temp.astype(np.float64))
                gain = current_max - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
                    best_new_h = h_temp
        
        if best_gain > 1e-6:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Binary Local Search
    for restart in range(20):
        if time.time() > deadline - 20:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # 2. Continuous Refinement
    # Use the best binary solution as a starting point for continuous optimization
    # Objective: minimize max correlation
    # This is non-smooth. SLSQP might work if we use a smooth approximation or just rely on it handling the max.
    # Actually, maximizing a sum is smooth, but we are minimizing the MAX of sums.
    # We can use a soft-max approximation: log(sum(exp(lambda * C[k])))/lambda
    # Or just use the previous SLSQP approach but start from binary.
    
    def objective_soft_max(x):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        # Soft max
        # Scale corr to avoid overflow/underflow issues
        # corr values are around 10-30
        lam = 10.0
        # Shift
        shifted = corr - np.max(corr)
        soft_max = np.log(np.sum(np.exp(lam * shifted))) / lam + np.max(corr)
        return soft_max

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective_soft_max, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate
        except Exception:
            pass

    return best_h.tolist()
