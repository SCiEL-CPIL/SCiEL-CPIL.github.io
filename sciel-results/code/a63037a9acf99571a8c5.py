"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step function on [-1/4, 1/4]."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    delta = 0.5 / n  # interval width
    
    # Discrete norms (continuous approximations)
    l1 = delta * np.sum(h)
    l2_sq = delta * np.sum(h**2)
    
    if l1 < 1e-30:
        return 0.0
    
    # Compute autoconvolution supremum
    # For step functions, the autoconvolution at lag k*delta (k in integers) can be computed
    # The autoconvolution is piecewise linear, peaks at integer lags
    # (h*h)(k*delta) = delta * sum_i h_i * h_{i+k} for the "aligned" part
    # But actually for continuous step functions, we need to be more careful.
    # 
    # For a step function h(x) = sum_i h_i * 1_{[a_i, a_i+delta)}(x),
    # (h*h)(t) = sum_{i,j} h_i * h_j * max(0, delta - |t - (a_i - a_j)|)
    # 
    # The maximum occurs at t = a_i - a_j for some i,j, giving delta * sum_{i,j: a_i-a_j=t} h_i*h_j
    # At t=0: delta * sum_i h_i^2
    # At t = k*delta: delta * sum_i h_i * h_{i+k}
    # 
    # For nonnegative h, the max is at t=0 by Cauchy-Schwarz on the discrete version.
    # But let's verify numerically for the step function case.
    
    # Compute discrete autoconvolution at all integer lags
    # (h*h)_k = sum_i h_i * h_{i+k} for k = -(n-1), ..., n-1
    conv = np.correlate(h, h, mode='full')
    # The continuous autoconvolution at lag k*delta is delta * conv[k + n - 1]
    # The supremum of the continuous autoconvolution is the max of this (since it's piecewise linear
    # with peaks at integer lags for nonnegative step functions)
    conv_sup = delta * np.max(conv)
    
    if conv_sup < 1e-30:
        return 0.0
    
    return l2_sq / (l1 * conv_sup)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = N
    best_h = None
    best_ratio = 0.0
    
    # Strategy: Use a sparse set of indices (Sidon-like) and optimize amplitudes
    # Start with various sparse patterns
    
    # Pattern 1: All ones
    h = np.ones(n)
    r = _compute_ratio(h)
    best_h = h.copy()
    best_ratio = r
    
    # Pattern 2: Sparse - every k-th index
    for k in [2, 3, 4, 5, 7, 10]:
        h = np.zeros(n)
        h[::k] = 1.0
        r = _compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Pattern 3: Power of 2 positions
    h = np.zeros(n)
    pos = 1
    while pos < n:
        h[pos] = 1.0
        pos *= 2
    r = _compute_ratio(h)
    if r > best_ratio:
        best_ratio = r
        best_h = h.copy()
    
    # Pattern 4: Fibonacci-like spacing
    h = np.zeros(n)
    fibs = [0, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    for f in fibs:
        if f < n:
            h[f] = 1.0
    r = _compute_ratio(h)
    if r > best_ratio:
        best_ratio = r
        best_h = h.copy()
    
    # Now try to optimize amplitudes using gradient ascent on log-ratio
    # for the best pattern found so far
    if time.time() < deadline - 30:
        h_opt = best_h.copy()
        # Use simple gradient ascent
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 20:
                break
            h_eval = h_opt.copy()
            r = _compute_ratio(h_eval)
            if r <= 0:
                break
            
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros(n)
            for i in range(n):
                h_plus = h_opt.copy()
                h_plus[i] += eps
                h_minus = h_opt.copy()
                h_minus[i] -= eps
                if h_minus[i] < 0:
                    h_minus[i] = 0
                r_plus = _compute_ratio(h_plus)
                r_minus = _compute_ratio(h_minus)
                grad[i] = (r_plus - r_minus) / (2 * eps)
            
            h_new = h_opt + lr * grad
            h_new = np.maximum(h_new, 0)
            
            r_new = _compute_ratio(h_new)
            if r_new > r:
                h_opt = h_new
                if r_new > best_ratio:
                    best_ratio = r_new
                    best_h = h_new.copy()
            else:
                lr *= 0.9
        
        # Normalize to avoid numerical issues
        if np.max(best_h) > 0:
            best_h = best_h / np.max(best_h)
    
    # Final check
    r_final = _compute_ratio(best_h)
    if r_final <= 0:
        return [1.0] * n
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
