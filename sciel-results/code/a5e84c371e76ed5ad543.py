import numpy as np
from scipy.optimize import minimize
import time

def compute_P(support, weights):
    n = len(support)
    # support: (n,), weights: (n,)
    # P = sum_{k,l} w_k w_l * (sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k])
    
    # Vectorized computation
    # s_i + s_j: (n, n)
    sums_ij = support[:, None] + support[None, :]
    # 2*s_l - s_k: (n, n) where index 0 is k, index 1 is l?
    # Let's align indices carefully.
    # k is index for X3, l is index for X4.
    # Condition: s_i + s_j < 2*s_l - s_k
    # RHS depends on k, l.
    # We need sum over i, j.
    
    # This is a 4D sum. n=10 -> 10^4 = 10000 ops. Very fast.
    # n=20 -> 20^4 = 160000 ops. Still fast.
    
    P = 0.0
    # To be efficient, we can precompute the "mass" of sums less than a threshold.
    # But n is small, so direct loop or vectorization is fine.
    
    # Vectorization:
    # Create meshgrid for k, l
    k_idx, l_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    # Thresholds T[k,l] = 2*s[l] - s[k]
    thresholds = 2 * support[l_idx] - support[k_idx]
    
    # For each (k,l), we need sum_{i,j} w_i w_j * 1[s_i+s_j < T[k,l]]
    # This is (w * (s_i+s_j < T)) @ w ? No, it's sum_i sum_j w_i w_j I(...)
    # Let M_T be matrix of indicators I[s_i+s_j < T].
    # Sum is w^T M_T w.
    
    # We can do this by sorting sums_ij and using cumulative sums?
    # Or just loop over k,l since n is small.
    
    total = 0.0
    w = weights / np.sum(weights) # normalize
    
    for k in range(n):
        for l in range(n):
            T = 2 * support[l] - support[k]
            # sum_{i,j} w_i w_j * 1[s_i+s_j < T]
            # Using broadcasting
            mask = sums_ij < T
            # Sum of w_i w_j over mask
            # This is (w * mask.sum(axis=1)) @ w? No.
            # It is sum_i (w_i * sum_j (w_j * mask[i,j]))
            inner = (w * mask).sum(axis=1) # sum_j w_j mask[i,j] for each i
            val = (w * inner).sum()
            total += w[k] * w[l] * val
            
    return total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600
    best_score = 0.0
    best_support = None
    best_weights = None
    
    # Try different n
    for n in range(2, 11):
        if time.time() > deadline - 60:
            break
            
        # Initial guess: spread points in [0, 1]
        s_init = np.linspace(0, 1, n)
        w_init = np.ones(n) / n
        
        # Variables: s_1...s_{n-1} (fix s_0=0, s_{n-1}=1? Or just fix s_{n-1}=1 and s_0>=0)
        # Let's fix s[0]=0 and s[-1]=1 to remove scale and shift ambiguity?
        # Shift is not invariant. Scale is invariant.
        # So fix s_max = 1.
        # We can let s_0 be free in [0,1].
        
        def objective(vars):
            if time.time() > deadline - 10:
                return -1.0 # Timeout penalty
            
            # Unpack
            # vars: s[0..n-2], w[0..n-1]
            s = np.zeros(n)
            s[0:n-1] = vars[0:n-1]
            s[n-1] = 1.0
            
            # Ensure sorted? Not strictly required for compute_P, but good for stability.
            # However, if we don't sort, the indices in w must match s.
            # It's better to sort s and w together?
            # If we optimize s, they might cross.
            # Let's enforce s[i] <= s[i+1] by using a monotonic transformation?
            # Or just sort them after extraction. But then w must be permuted accordingly.
            # This makes the objective non-smooth.
            
            # Alternative: optimize s directly and allow them to be unsorted.
            # The formula doesn't require sorted s.
            
            w_raw = vars[n-1:2*n-1]
            # Softmax for weights to ensure positivity and sum 1
            w = np.exp(w_raw - np.max(w_raw))
            w = w / np.sum(w)
            
            # Ensure s >= 0
            s = np.maximum(s, 0.0)
            
            # Check validity
            if np.any(np.isnan(s)) or np.any(np.isnan(w)):
                return -1.0
            
            P = compute_P(s, w)
            return -P
        
        # Initial vars
        s_init_vars = s_init[0:n-1]
        w_init_vars = np.zeros(n) # log weights
        
        # Optimize
        res = minimize(objective, np.concatenate([s_init_vars, w_init_vars]), method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < 0:
            P_val = -res.fun
            if P_val > best_score:
                best_score = P_val
                s_final = np.zeros(n)
                s_final[0:n-1] = res.x[0:n-1]
                s_final[n-1] = 1.0
                s_final = np.maximum(s_final, 0.0)
                
                w_raw = res.x[n-1:2*n-1]
                w_final = np.exp(w_raw - np.max(w_raw))
                w_final = w_final / np.sum(w_final)
                
                best_support = s_final
                best_weights = w_final

    if best_support is None:
        # Fallback to seed
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
        
    return best_support, best_weights
