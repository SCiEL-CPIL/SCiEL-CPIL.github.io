import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def generate_sidon_set(n, max_size):
    """Generate a Sidon set of size max_size in [0, n-1]."""
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _ratio(best_h)
    
    # Try different Sidon set sizes
    for size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        sidon_set = generate_sidon_set(N, size)
        if len(sidon_set) < size:
            continue
            
        # Initialize with uniform heights on the Sidon set
        h0 = np.zeros(N)
        for idx in sidon_set:
            h0[idx] = 1.0
        
        # Optimize amplitudes
        # We optimize on the active indices only to reduce dimensionality
        active_indices = np.array(sidon_set)
        
        def neg_ratio_sparse(x):
            h_full = np.zeros(N)
            h_full[active_indices] = np.maximum(x, 0)
            return -_ratio(h_full)
        
        x0 = np.ones(len(active_indices))
        
        try:
            result = minimize(neg_ratio_sparse, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            x_opt = np.maximum(result.x, 0)
            h_candidate = np.zeros(N)
            h_candidate[active_indices] = x_opt
            
            sc = _ratio(h_candidate)
            if sc > best_score:
                best_score = sc
                best_h = h_candidate.copy()
                
        except Exception:
            continue

    # Also try random sparse patterns
    np.random.seed(42)
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        k = np.random.randint(5, 15)
        indices = np.random.choice(N, k, replace=False)
        h0 = np.zeros(N)
        h0[indices] = np.random.uniform(0.5, 1.5, k)
        
        active_indices = indices
        def neg_ratio_rand(x):
            h_full = np.zeros(N)
            h_full[active_indices] = np.maximum(x, 0)
            return -_ratio(h_full)
            
        try:
            result = minimize(neg_ratio_rand, h0[indices], method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            x_opt = np.maximum(result.x, 0)
            h_candidate = np.zeros(N)
            h_candidate[active_indices] = x_opt
            sc = _ratio(h_candidate)
            if sc > best_score:
                best_score = sc
                best_h = h_candidate.copy()
        except Exception:
            continue

    return best_h.tolist()
