"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines linear overlap definition (FFT-accelerated) with analytical gradient optimization.
Structural change: uses vectorized gradient and epigraph formulation for SLSQP."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    # r_k = sum_{i=k}^{n-1} h[i] * h[i-k] for k=0..n-1
    # This is the autocorrelation. Use FFT for O(n log n) instead of O(n^2).
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # Autocorrelation: ifft(|H|^2) gives circular autocorrelation of padded signal
    # For linear autocorrelation of length-n signal, we need:
    # r[k] = sum_{i=0}^{n-1-k} h[i+k] * h[i] for k=0..n-1
    # Using FFT: correlate h with h
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # corr[k] for k=0..n-1 gives r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max(h, beta):
    """
    Smooth maximum of the linear overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow in exp
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = (2/n)*(n/2 - r_k) where r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    # So grad_j = -(2/n) * sum_k w_k * (h_{j-k} + h_{j+k})
    
    # Vectorized: for each j, grad_j = -(2/n) * [sum_{k=0}^{j} w_k * h_{j-k} + sum_{k=1}^{n-1-j} w_k * h_{j+k}]
    # This is a convolution-like operation.
    
    # Term 1: sum_{k=0}^{j} w_k * h_{j-k} = (w * h)[j] for j=0..n-1 (linear correlation)
    # Term 2: sum_{k=1}^{n-1-j} w_k * h_{j+k} = (w * h)[j] with different indexing
    
    # Use np.convolve for the first term (full convolution, take appropriate part)
    # (w * h)[j] = sum_{k=0}^{j} w[k] * h[j-k] for j < n
    conv1 = np.convolve(w, h, mode='full')[:n]
    
    # For term 2: sum_{k=1}^{n-1-j} w[k] * h[j+k]
    # Let i = j+k, so i goes from j+1 to n-1, k = i-j
    # = sum_{i=j+1}^{n-1} w[i-j] * h[i]
    # This is like a cross-correlation. We can compute it as:
    # For each j, sum_{k=1}^{n-1-j} w[k] * h[j+k]
    # = sum_{k=0}^{n-1-j-1} w[k+1] * h[j+k+1]
    # Let's use another convolution approach
    # w_rev = w[::-1], then convolve w_rev with h
    # (w_rev * h)[m] = sum_k w_rev[k] * h[m-k] = sum_k w[n-1-k] * h[m-k]
    # We want sum_{k=1}^{n-1-j} w[k] * h[j+k]
    # Let l = k, so sum_{l=1}^{n-1-j} w[l] * h[j+l]
    # = sum_{l=0}^{n-2-j} w[l+1] * h[j+l+1]
    
    # Alternative: use the fact that this is the "other side" of the autocorrelation
    # Let's just compute it directly but more efficiently using numpy operations
    # For each j: sum_{k=1}^{n-1-j} w[k] * h[j+k]
    # We can vectorize by creating a matrix, but that's O(n^2) memory.
    # Instead, use the symmetry: note that the full expression for grad_j is:
    # -(2/n) * [sum_{k=0}^{min(j,n-1)} w[k]*h[j-k] + sum_{k=1}^{n-1-j} w[k]*h[j+k]]
    # But actually, let me reconsider. The original loop is O(n^2) but with small constants.
    # For n=100, that's 10000 operations per gradient call, which should be fine.
    # Let me keep the loop but optimize it slightly.
    
    # Actually, let's use a cleaner vectorized approach with stride tricks
    # Create an array where we can compute both sums at once
    # For term 1: grad1[j] = sum_{k=0}^{j} w[k] * h[j-k] = conv(w, h)[j]
    # For term 2: grad2[j] = sum_{k=1}^{n-1-j} w[k] * h[j+k]
    
    # For term 2, note that if we define w2 = w[1:], then
    # grad2[j] = sum_{k=0}^{n-2-j} w2[k] * h[j+1+k]
    # This is like a shifted correlation
    
    # Let me just use the loop but with numpy vectorization inside
    for k in range(n):
        wk = w[k]
        if wk == 0.0:
            continue
        factor = -(2.0 / n) * wk
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Start: Cosine Wave
    # h_i = 0.5 + 0.5 * cos(2 * pi * i / N)
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Optimization with Analytical Gradient
    # Use SLSQP with the smooth max objective and its gradient
    
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Refine Cosine Start
    if time.time() - t0 < _DEADLINE - 50:
        _run_optimization(h_cos, beta=50.0, maxiter=300, ftol=1e-10)
        
    # Phase 2: Try diverse structured starts
    if time.time() - t0 < _DEADLINE - 40:
        # Sine wave
        h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
        h_sin = _fix_sum(h_sin)
        _update_best(h_sin)
        _run_optimization(h_sin, beta=50.0, maxiter=200, ftol=1e-9)
        
    if time.time() - t0 < _DEADLINE - 30:
        # Triangle wave
        h_tri = 1.0 - 2.0 * np.abs((i % N) / N - 0.5)
        h_tri = _fix_sum(h_tri)
        _update_best(h_tri)
        _run_optimization(h_tri, beta=50.0, maxiter=200, ftol=1e-9)
        
    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=150, ftol=1e-9)
            
    # Phase 4: Final Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=100.0, maxiter=300, ftol=1e-12)
        
    # Phase 5: Ultra-high precision polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        _run_optimization(best_h, beta=200.0, maxiter=200, ftol=1e-14)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
