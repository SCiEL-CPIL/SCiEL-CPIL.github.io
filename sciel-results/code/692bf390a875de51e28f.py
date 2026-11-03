"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated linear overlap computation with analytical gradient-based
optimization using a smooth-max (log-sum-exp) surrogate.
Includes multi-phase optimization with increasing beta for refinement and polishing.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
# Safety margin: Evaluator kills at 150s. We target finishing by 140s.
_DEADLINE = time.time() + 140


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
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


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    
    # Numerical stability for log-sum-exp
    # beta * (cs - m) <= 0
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # Fallback if underflow (unlikely with m subtraction)
        w = np.full(n, 1.0/n)
        val = m
    else:
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient computation
    # To speed up slightly, we can skip negligible weights, but N=100 is small.
    for k in range(n):
        if w[k] < 1e-12:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
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
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
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


def _run_slsqp(h0, beta, maxiter, ftol):
    """
    Run SLSQP optimization with smooth max objective.
    Returns the optimized array or the input if it fails/times out.
    """
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
        if res.x is not None:
            return res.x
        return h0
    except Exception:
        return h0


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
    
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Constant
    h_const = np.full(N, 0.5)
    _update_best(h_const)
    
    # Sine
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    _update_best(h_alt)
    
    # Random
    h_rand = np.random.uniform(0.1, 0.9, N)
    _update_best(h_rand)
    
    # 2. Optimization Phases
    
    # Phase 1: Broad search with moderate beta
    # We try to optimize from the best initial candidate and a few others
    candidates = [h_cos, h_const, h_sin, h_alt, h_rand]
    
    for h0 in candidates:
        if time.time() - t0 > _DEADLINE - 20:
            break
        # Moderate beta for smooth landscape
        h_opt = _run_slsqp(h0, beta=30.0, maxiter=200, ftol=1e-8)
        _update_best(h_opt)
        
    # Phase 2: Refinement with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-10)
        _update_best(h_opt)
        
    # Phase 3: Final Polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        # High beta makes the objective very sharp, requiring tighter tolerance
        h_opt = _run_slsqp(best_h, beta=300.0, maxiter=500, ftol=1e-12)
        _update_best(h_opt)
        
    # Phase 4: Emergency polish if time remains
    if best_h is not None and time.time() - t0 < _DEADLINE - 0.5:
        h_opt = _run_slsqp(best_h, beta=500.0, maxiter=100, ftol=1e-13)
        _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
