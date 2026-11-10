import numpy as np
from scipy.optimize import minimize
import time

def compute_P(s, w):
    """
    Computes P[X1 + X2 + X3 < 2*X4] efficiently.
    s: support points (n,)
    w: weights (n,)
    """
    n = len(s)
    if n == 0:
        return 0.0
        
    # Pairwise sums S_ij = s_i + s_j
    S_ij = s[:, np.newaxis] + s[np.newaxis, :]
    # Pairwise weight products W_ij = w_i * w_j
    W_ij = w[:, np.newaxis] * w[np.newaxis, :]
    
    # Flatten and sort
    S_flat = S_ij.flatten()
    W_flat = W_ij.flatten()
    
    # Sort by S_ij
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    
    # Cumulative sum of weights
    W_cumsum = np.cumsum(W_sorted)
    
    P = 0.0
    for l in range(n):
        # Thresholds T_k = 2*s_l - s_k for all k
        T_k = 2 * s[l] - s
        # Find number of pairs (i,j) such that S_ij < T_k
        # np.searchsorted returns index of first element >= T_k
        # So index is the count of elements < T_k
        counts = np.searchsorted(S_sorted, T_k, side='left')
        # The sum of W_ij for S_ij < T_k is W_cumsum[counts - 1]
        # Need to handle counts == 0
        Q_l = np.zeros(n)
        mask = counts > 0
        Q_l[mask] = W_cumsum[counts[mask] - 1]
        
        # P_l = sum_k w_k * Q_l[k]
        P_l = np.dot(w, Q_l)
        P += w[l] * P_l
        
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590  # Keep 10s margin
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    # Try different n values
    for n in [20, 30]:
        if time.time() > deadline - 60:
            break
            
        # Initial support: uniform on [0, 1]
        s = np.linspace(0, 1, n)
        # Initial weights: uniform
        w = np.ones(n) / n
        
        # Local best for this n
        local_best_P = compute_P(s, w)
        local_best_s = s.copy()
        local_best_w = w.copy()
        
        # Optimization loop for this n
        # We will alternate optimizing w and s
        # Max iterations or time per n
        max_iter = 20
        for _ in range(max_iter):
            if time.time() > deadline - 30:
                break
                
            # Optimize weights
            def objective_w(params):
                # params are weights, sum to 1, >= 0
                return -compute_P(local_best_s, params)
                
            # Optimize support
            def objective_s(params):
                # params are support, >= 0
                return -compute_P(params, local_best_w)
            
            # SLSQP for weights
            res_w = minimize(
                objective_w, 
                local_best_w, 
                method='SLSQP', 
                bounds=[(0, 1)] * n,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            if res_w.success:
                w_new = res_w.x
                # Clip small negatives
                w_new = np.maximum(w_new, 0)
                w_new = w_new / np.sum(w_new)
                P_new = compute_P(local_best_s, w_new)
                if P_new > local_best_P:
                    local_best_P = P_new
                    local_best_w = w_new.copy()
            
            # SLSQP for support
            res_s = minimize(
                objective_s, 
                local_best_s, 
                method='SLSQP', 
                bounds=[(0, 10)] * n, # Upper bound 10 to prevent explosion
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            if res_s.success:
                s_new = res_s.x
                s_new = np.maximum(s_new, 0)
                P_new = compute_P(s_new, local_best_w)
                if P_new > local_best_P:
                    local_best_P = P_new
                    local_best_s = s_new.copy()
                    
        # Update global best
        if local_best_P > best_P:
            best_P = local_best_P
            best_s = local_best_s.copy()
            best_w = local_best_w.copy()
            
    # Fallback if no valid result found
    if best_s is None:
        n = 32
        best_s = np.arange(n, dtype=np.float64)
        best_w = np.ones(n, dtype=np.float64) / n
        
    return best_s, best_w
