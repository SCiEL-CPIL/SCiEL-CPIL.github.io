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

def _correlations(h):
    h = np.asarray(h, dtype=np.float64)
    return np.correlate(h, 1.0 - h, mode="full")

def _epigraph_objective(x):
    # x = [h_0, ..., h_{N-1}, t]
    h = x[:N]
    t = x[N]
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = _correlations(h_clipped)
    # Objective: t (we minimize t, constrained by corr <= t)
    # SLSQP minimizes the objective subject to constraints
    # We want to minimize t.
    return t

def _epigraph_constraints(x):
    # x = [h_0, ..., h_{N-1}, t]
    h = x[:N]
    t = x[N]
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = _correlations(h_clipped)
    
    # Constraints: corr[k] <= t for all k
    # In SLSQP, constraints are of the form fun(x) >= 0
    # So we define fun(x) = t - corr[k] >= 0
    cons = []
    for k in range(len(corr)):
        cons.append({'type': 'ineq', 'fun': lambda x, k=k: x[N] - _correlations(np.clip(x[:N], 0.0, 1.0))[k]})
    
    # Equality constraint: sum(h) = N/2
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N/2.0})
    
    return cons

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Binary Local Search
    for restart in range(20):
        if time.time() > deadline - 30:
            break
            
        h_bin = np.zeros(N, dtype=np.int8)
        ones_idx = rng.choice(N, size=N//2, replace=False)
        h_bin[ones_idx] = 1
        
        h_opt, max_corr = _local_search_binary(h_bin)
        val = max_corr / N * 2.0
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
            
    # 2. Continuous Refinement using Epigraph Formulation
    # Variables: h (N), t (1)
    # Minimize t
    # Subject to:
    #   corr_k(h) <= t for all k
    #   sum(h) = N/2
    #   0 <= h_i <= 1
    
    # Initial guess for t: current max corr + small margin
    init_corr = _correlations(best_h)
    init_t = np.max(init_corr) + 1.0
    
    x0 = np.concatenate([best_h, [init_t]])
    
    bounds = [(0.0, 1.0)] * N + [(0.0, np.inf)]
    
    if time.time() < deadline - 5:
        try:
            # Use SLSQP
            # We need to define the constraints carefully.
            # Defining lambda functions inside the list is slow and potentially buggy if not careful with closures.
            # Let's define a helper function to return the list of constraint dicts.
            
            def get_constraints():
                cons = []
                # Inequality constraints: t - corr_k >= 0
                # We can vectorize this if we use a single constraint function that returns a vector?
                # SLSQP expects a list of dicts, or a single dict with 'fun' returning a vector for 'ineq'.
                # Let's use the vectorized approach for efficiency.
                
                def ineq_fun(x):
                    h = np.clip(x[:N], 0.0, 1.0)
                    t = x[N]
                    corr = _correlations(h)
                    return t - corr
                    
                def eq_fun(x):
                    return np.sum(x[:N]) - N/2.0
                
                cons.append({'type': 'ineq', 'fun': ineq_fun})
                cons.append({'type': 'eq', 'fun': eq_fun})
                return cons
            
            res = minimize(
                _epigraph_objective, 
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=get_constraints(),
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < init_t:
                h_candidate = np.clip(res.x[:N], 0.0, 1.0)
                # Fix sum to be exactly N/2
                diff = N/2.0 - np.sum(h_candidate)
                if abs(diff) > 1e-9:
                    # Simple projection: add diff/N to all, then clip
                    # This might violate bounds, so we might need a more careful projection
                    # But for small diffs, it should be fine.
                    h_candidate = h_candidate + diff / N
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                    
                    # Second pass to fix sum exactly if clipping changed it
                    diff2 = N/2.0 - np.sum(h_candidate)
                    if abs(diff2) > 1e-9:
                         # Adjust only those not at bounds
                         # This is a heuristic fix
                         indices = np.where((h_candidate > 1e-9) & (h_candidate < 1.0 - 1e-9))[0]
                         if len(indices) > 0:
                             h_candidate[indices] += diff2 / len(indices)
                             h_candidate = np.clip(h_candidate, 0.0, 1.0)
                 
                val = _value(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate
        except Exception:
            pass

    return best_h.tolist()
