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
    
    # Use a grid-based approach with larger n for better resolution
    n = 64
    s = np.linspace(0, 1, n)
    w = np.ones(n) / n
    
    local_best_P = compute_P(s, w)
    local_best_s = s.copy()
    local_best_w = w.copy()
    
    if local_best_P > best_P:
        best_P = local_best_P
        best_s = local_best_s.copy()
        best_w = local_best_w.copy()
    
    # Simulated annealing / random perturbation approach
    # We'll try to improve by perturbing weights and support
    
    rng = np.random.default_rng(42)
    T = 0.1  # Temperature for simulated annealing
    T_min = 0.001
    cooling = 0.95
    
    # Precompute for faster evaluation
    max_iter = 200
    
    for iter in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        # Randomly choose what to perturb: weights or support
        if rng.random() < 0.5:
            # Perturb weights
            w_new = local_best_w.copy()
            # Add random noise
            noise_scale = 0.1 * T
            w_new += rng.normal(0, noise_scale, size=n)
            # Ensure non-negative and normalize
            w_new = np.maximum(w_new, 0)
            w_sum = np.sum(w_new)
            if w_sum > 1e-9:
                w_new = w_new / w_sum
                P_new = compute_P(local_best_s, w_new)
                delta = P_new - local_best_P
                if delta > 0 or rng.random() < np.exp(delta / max(T, T_min)):
                    local_best_P = P_new
                    local_best_w = w_new.copy()
                    if local_best_P > best_P:
                        best_P = local_best_P
                        best_s = local_best_s.copy()
                        best_w = local_best_w.copy()
        else:
            # Perturb support
            s_new = local_best_s.copy()
            noise_scale = 0.05 * T
            s_new += rng.normal(0, noise_scale, size=n)
            # Ensure non-negative
            s_new = np.maximum(s_new, 0)
            P_new = compute_P(s_new, local_best_w)
            delta = P_new - local_best_P
            if delta > 0 or rng.random() < np.exp(delta / max(T, T_min)):
                local_best_P = P_new
                local_best_s = s_new.copy()
                if local_best_P > best_P:
                    best_P = local_best_P
                    best_s = local_best_s.copy()
                    best_w = local_best_w.copy()
                    
        # Cool down
        T = max(T * cooling, T_min)
        
        # Every 20 iterations, try a more aggressive restart
        if (iter + 1) % 20 == 0 and time.time() < deadline - 30:
            # Try a different initial distribution
            s_try = np.linspace(0, 1, n)
            # Try various weight distributions
            for exp_val in [1.5, 2.0, 2.5, 3.0]:
                x = np.linspace(0, 1, n)
                w_try = x ** exp_val
                w_try_sum = np.sum(w_try)
                if w_try_sum > 1e-9:
                    w_try = w_try / w_try_sum
                    P_try = compute_P(s_try, w_try)
                    if P_try > local_best_P:
                        local_best_P = P_try
                        local_best_s = s_try.copy()
                        local_best_w = w_try.copy()
                        if local_best_P > best_P:
                            best_P = local_best_P
                            best_s = local_best_s.copy()
                            best_w = local_best_w.copy()
                            break
    
    # Fallback if no valid result found
    if best_s is None:
        best_s = np.arange(32, dtype=np.float64)
        best_w = np.ones(32, dtype=np.float64) / 32
        
    return best_s, best_w
