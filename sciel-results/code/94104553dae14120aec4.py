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
                    best_new_h = h_temp
        
        if best_gain > 1e-6 and best_new_h is not None:
            h_bin = best_new_h
            current_max = current_max - best_gain
            ones = np.where(h_bin == 1)[0]
            zeros = np.where(h_bin == 0)[0]
            improved = True
            
    return h_bin.astype(np.float64), current_max

def _continuous_optimize(h_start, deadline):
    best_h = h_start.copy()
    best_val = _value(best_h)
    
    # Objective: max corr
    # We minimize t subject to C_k <= t
    # Or just minimize max(C_k) directly using a smooth approximation?
    # Let's use the constraint method.
    
    def objective(x):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        return np.max(corr)
        
    # This is non-smooth.
    # Let's use SLSQP with the constraint formulation.
    
    # Variables: h
    # Minimize t
    # Subject to C_k - t <= 0, sum(h)=50, 0<=h<=1
    
    # To avoid the complexity of t, we can just minimize the sum of squares of (C_k - t) + t?
    # No, let's stick to the previous convex refine but make it faster.
    
    # Actually, let's try a simple gradient descent on the max function using subgradients.
    # Or just use `minimize` with method='SLSQP' on the raw max function?
    # SLSQP can handle non-smooth functions poorly.
    
    # Let's use the trust-constr approach from the current program but optimized.
    
    from scipy.optimize import NonlinearConstraint
    
    num_corr = 2 * N - 1
    
    def constraint_func(x):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        # We want to minimize max corr.
        # We can't easily constrain max corr without t.
        # Let's just minimize the sum of top 10 corrs?
        # Or just minimize the max using a penalty.
        
        # Let's try minimizing the max directly with a smooth approximation.
        # max(x) approx sum(x^p)^(1/p)
        # This is hard to optimize.
        
        # Let's go back to the t variable.
        pass
        
    # The previous _convex_refine was correct but slow.
    # I will implement a faster version using L-BFGS-B on the penalized objective.
    
    # Objective: sum(C_k^2) + lambda * max(C_k)
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Binary local search
    for restart in range(10):
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

    # Continuous refinement
    # Use SLSQP to minimize max corr
    # We will define a function that returns the max corr and use it in minimize.
    # Since max is non-smooth, we will use a multi-start approach.
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Start from best_h or random
        if restart == 0:
            h_start = best_h.copy()
        else:
            h_start = np.clip(best_h + rng.normal(0, 0.2, size=N), 0.0, 1.0)
            h_start += (N/2.0 - np.sum(h_start)) / N
            h_start = np.clip(h_start, 0.0, 1.0)
            
        def obj(x):
            h = np.clip(x, 0.0, 1.0)
            corr = np.correlate(h, 1.0 - h, mode="full")
            return np.max(corr)
            
        def jac(x):
            h = np.clip(x, 0.0, 1.0)
            corr = np.correlate(h, 1.0 - h, mode="full")
            max_idx = np.argmax(corr)
            # Gradient of C_k
            # C_k = sum_i h_i (1-h_{i+k-(N-1)})
            # d/dh_j C_k = (1-h_{j+k-(N-1)}) - h_{j-k+(N-1)}
            shift = max_idx - (N-1)
            g = np.zeros(N)
            for j in range(N):
                term1 = 1.0
                if 0 <= j + shift < N:
                    term1 -= h[j + shift]
                term2 = 0.0
                if 0 <= j - shift < N:
                    term2 = h[j - shift]
                g[j] = term1 - term2
            return g
            
        try:
            res = minimize(obj, h_start, method='L-BFGS-B', jac=jac,
                           bounds=[(0,1)]*N,
                           options={'maxiter': 100, 'gtol': 1e-6})
            h_cand = np.clip(res.x, 0.0, 1.0)
            h_cand += (N/2.0 - np.sum(h_cand)) / N
            h_cand = np.clip(h_cand, 0.0, 1.0)
            val = _value(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand
        except Exception:
            pass

    return best_h.tolist()
