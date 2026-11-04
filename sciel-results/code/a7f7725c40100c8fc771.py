import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _neg_ratio(params):
    h = np.exp(params)
    return -_compute_ratio(h)

def _get_sidon_set(n, max_val):
    """Generate a Sidon set of size n within [0, max_val]."""
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        new_sums = set()
        valid = True
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def _optimize_weights(sidon_indices, time_budget=10.0):
    """Optimize weights on Sidon indices to maximize the ratio."""
    k = len(sidon_indices)
    if k == 0:
        return np.zeros(N)
    
    start = time.time()
    
    # Parameterize: log-weights for positivity, with L1 normalization
    # Use softmax-like parameterization: w_i = exp(x_i) / sum(exp(x_j))
    def _objective(x):
        w = np.exp(x - np.max(x))  # numerical stability
        w = w / np.sum(w)
        h = np.zeros(N)
        for i, idx in enumerate(sidon_indices):
            h[idx] = w[i]
        return -_compute_ratio(h)
    
    # Start from uniform weights
    x0 = np.zeros(k)
    
    # Try multiple random restarts
    best_x = x0.copy()
    best_val = _objective(x0)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start > time_budget:
            break
        if r == 0:
            x_r = x0.copy()
        else:
            x_r = np.random.randn(k) * 0.5
        
        try:
            res = minimize(_objective, x_r, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Convert optimized weights to h
    w = np.exp(best_x - np.max(best_x))
    w = w / np.sum(w)
    h = np.zeros(N)
    for i, idx in enumerate(sidon_indices):
        h[idx] = w[i]
    return h

def _try_larger_sidon(max_size=15, time_budget=5.0):
    """Try to find a larger Sidon set and optimize on it."""
    start = time.time()
    best_h = None
    best_ratio = 0.0
    
    for size in range(8, max_size + 1):
        if time.time() - start > time_budget:
            break
        sidon = _get_sidon_set(size, N - 1)
        if len(sidon) < size:
            continue
        h = _optimize_weights(sidon, time_budget=max(1.0, time_budget - (time.time() - start)))
        r = _compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    return best_h, best_ratio

def construct():
    start = time.time()
    
    # Baseline: Sidon set of size 10 with uniform weights
    sidon_indices = _get_sidon_set(10, N - 1)
    h = np.zeros(N)
    for idx in sidon_indices:
        h[idx] = 1.0
    l1 = np.sum(h)
    if l1 > 0:
        h = h / l1
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    
    # Try optimizing weights on the same Sidon set
    if time.time() - start < _DEADLINE - 30:
        h_opt = _optimize_weights(sidon_indices, time_budget=20.0)
        r = _compute_ratio(h_opt)
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt.copy()
    
    # Try larger Sidon sets
    if time.time() - start < _DEADLINE - 20:
        h_large, r_large = _try_larger_sidon(max_size=14, time_budget=40.0)
        if h_large is not None and r_large > best_ratio:
            best_ratio = r_large
            best_h = h_large.copy()
    
    # Final: ensure nonnegative and valid
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        # Fallback to uniform
        best_h = np.ones(N) / N
    
    return [float(x) for x in best_h]
