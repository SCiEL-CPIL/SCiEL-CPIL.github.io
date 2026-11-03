"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-accelerated overlap computation with a smooth-max (log-sum-exp) surrogate.
Multi-phase optimization: coarse-to-fine beta schedule, multiple starts, and
analytical structure exploitation.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
# Evaluator kills at 150s. Target finishing by 142s.
_DEADLINE = time.time() + 142


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Value and gradient of the smooth maximum (log-sum-exp) of C_k.
    Returns (value, gradient).
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    
    # Numerically stable log-sum-exp
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        w = np.full(n, 1.0 / n)
        val = m
    else:
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k where r_k = sum_{i=k}^{n-1} h_i h_{i-k}
    # dC_k/dh_j = -(2/n) * (h_{j-k} [if j>=k] + h_{j+k} [if j+k<n])
    
    for k in range(n):
        if w[k] < 1e-14:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]
    
    return val, grad


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
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
    """Run SLSQP optimization with smooth max objective."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_val_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_val_grad(x, beta)
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


def _try_analytical_form():
    """
    Try structured forms that might be near-optimal.
    Based on the observation that for the constant h=0.5, all C_k = 1.
    We want to reduce the max C_k below 1.
    
    The key insight: C_k = 1 - (2/N) * r_k, so minimizing max C_k 
    is equivalent to maximizing min r_k.
    
    Try cosine-like profiles with different frequencies.
    """
    candidates = []
    i = np.arange(N)
    
    # Cosine with fundamental frequency
    for amp in [0.3, 0.4, 0.45, 0.5, 0.55]:
        h = 0.5 + amp * np.cos(2.0 * np.pi * i / N)
        candidates.append(h)
    
    # Sine with fundamental frequency
    for amp in [0.3, 0.4, 0.45, 0.5]:
        h = 0.5 + amp * np.sin(2.0 * np.pi * i / N)
        candidates.append(h)
    
    # Second harmonic
    for amp in [0.2, 0.3, 0.4]:
        h = 0.5 + amp * np.cos(4.0 * np.pi * i / N)
        candidates.append(h)
    
    # Mix of harmonics
    h = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h)
    
    h = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) - 0.1 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h)
    
    # Square-wave-like
    h = np.where(i % 2 == 0, 0.7, 0.3)
    candidates.append(h)
    
    h = np.where(i % 2 == 0, 0.6, 0.4)
    candidates.append(h)
    
    # Triangular wave
    phase = 2.0 * np.pi * i / N
    h = 0.5 + (2.0 / np.pi) * np.arcsin(np.sin(phase)) * 0.4
    candidates.append(h)
    
    return candidates


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Generate initial candidates
    i = np.arange(N)
    candidates = []
    
    # Constant
    candidates.append(np.full(N, 0.5))
    
    # Analytical structured forms
    candidates.extend(_try_analytical_form())
    
    # Random starts
    for _ in range(8):
        h = np.random.uniform(0.05, 0.95, N)
        candidates.append(h)
    
    # Evaluate all candidates
    for h0 in candidates:
        _update_best(h0)
    
    # 2. Coarse-to-fine optimization
    # Phase 1: Low beta for broad landscape exploration
    beta_schedule_phase1 = [10.0, 20.0, 40.0]
    for beta in beta_schedule_phase1:
        if time.time() - t0 > _DEADLINE - 25:
            break
        if best_h is not None:
            h_opt = _run_slsqp(best_h, beta=beta, maxiter=150, ftol=1e-7)
            _update_best(h_opt)
    
    # Phase 2: Medium beta for refinement
    beta_schedule_phase2 = [80.0, 150.0]
    for beta in beta_schedule_phase2:
        if time.time() - t0 > _DEADLINE - 15:
            break
        if best_h is not None:
            h_opt = _run_slsqp(best_h, beta=beta, maxiter=250, ftol=1e-9)
            _update_best(h_opt)
    
    # Phase 3: High beta for polishing
    beta_schedule_phase3 = [300.0, 500.0, 800.0]
    for beta in beta_schedule_phase3:
        if time.time() - t0 > _DEADLINE - 5:
            break
        if best_h is not None:
            h_opt = _run_slsqp(best_h, beta=beta, maxiter=400, ftol=1e-11)
            _update_best(h_opt)
    
    # Phase 4: Final emergency polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _run_slsqp(best_h, beta=1000.0, maxiter=200, ftol=1e-12)
        _update_best(h_opt)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
