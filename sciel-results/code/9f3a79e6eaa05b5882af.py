import time
import math
import random

import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _autoconv_peak_ratio(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: N * max_k (sum_i h_i * h_{i+k}) / (sum_i h_i)^2
    For uniform, this equals 1.0.
    """
    h = np.asarray(heights, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Use FFT for fast convolution
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    max_conv = np.max(c)
    return N * max_conv / (total * total)

def _gradient(h):
    """
    Compute gradient of the objective function.
    Uses the fact that the max of convolution occurs at some lag,
    and the gradient with respect to h is the flipped h shifted by that lag.
    """
    h = np.maximum(h, 0.0)
    total = h.sum()
    if total <= 0:
        return np.zeros_like(h)
    
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    
    m_star = int(np.argmax(c))
    peak = float(c[m_star])
    
    # The convolution c[k] = sum_i h[i] * h[i + k - (N-1)]
    # d(c[m_star])/d(h[j]) = h[j - (m_star - (N-1))] if valid
    offset = m_star - (N - 1)
    g = np.zeros_like(h)
    lo = max(0, offset)
    hi = min(N, N + offset)
    if offset >= 0:
        g[lo:hi] = h[lo - offset:hi - offset]
    else:
        g[lo:hi] = h[lo - offset:hi - offset]
    
    # Quotient rule: d(peak/s^2) = (g*s^2 - peak*2*s*h) / s^4
    grad = (g * total * total - peak * 2.0 * total * h) / (total ** 4)
    return grad

def _projected_gradient_descent(h0, max_iters=2000, lr_init=0.01):
    """
    Projected gradient descent with adaptive step size for nonnegative heights.
    Preserves total mass approximately.
    """
    h = h0.copy()
    best_h = h.copy()
    best_val = _autoconv_peak_ratio(h)
    
    lr = lr_init
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        grad = _gradient(h)
        grad_norm = float(np.max(np.abs(grad))) + 1e-12
        
        # Take step
        h_new = h - lr * grad / grad_norm
        
        # Project to nonnegative
        h_new = np.maximum(h_new, 0.0)
        
        # Preserve total mass
        s_old = float(np.sum(h))
        s_new = float(np.sum(h_new))
        if s_new > 1e-9:
            h_new = h_new * (s_old / s_new)
        
        val = _autoconv_peak_ratio(h_new)
        
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new
            # Increase step size slightly if improving
            lr = min(lr * 1.02, 0.1)
        else:
            h = h_new
            # Decrease step size if not improving
            lr *= 0.95
    
    return best_h, best_val

def _cosine_parameterize(coeffs, N):
    """
    Parameterize heights as baseline + cosine series.
    coeffs[0] is baseline, coeffs[1:] are cosine coefficients.
    heights[i] = coeffs[0] + sum_{k=1}^{K} coeffs[k] * cos(2*pi*k*i/N)
    """
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, len(coeffs)):
        h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
    return h

def _score_from_coeffs(coeffs, N):
    h = _cosine_parameterize(coeffs, N)
    # Ensure nonnegativity by clipping
    h = np.maximum(h, 0)
    return _autoconv_peak_ratio(h)

def _simulated_annealing(h, max_iters=150, t0=1.0, t1=0.01):
    """Simulated annealing to fine-tune the profile."""
    h = h.copy()
    best_h = h.copy()
    best_val = _autoconv_peak_ratio(h)
    
    for it in range(max_iters):
        if _time_left() <= 5.0:
            break
        
        T = t0 * (t1 / t0) ** (it / max_iters)
        
        # Generate a random perturbation
        rng = np.random.RandomState(it * 12345 + 6789)
        h_new = h.copy()
        
        # Randomly perturb a few elements
        n_perturb = rng.randint(5, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            delta = rng.normal(0, 0.1 * np.mean(h))
            h_new[idx] = max(0.0, h_new[idx] + delta)
        
        # Preserve total mass approximately
        s_old = float(np.sum(h))
        s_new = float(np.sum(h_new))
        if s_new > 1e-9:
            h_new = h_new * (s_old / s_new)
        
        v_new = _autoconv_peak_ratio(h_new)
        
        # Accept if better, or with probability exp(-delta/T)
        if v_new < best_val:
            h = h_new
            best_h = h.copy()
            best_val = v_new
        else:
            delta = v_new - best_val
            if delta > 0 and rng.random() < math.exp(-delta / (T + 1e-10)):
                h = h_new
    
    return best_h, best_val

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _autoconv_peak_ratio(best_h)
    
    # --- Phase 1: Cosine series optimization with multiple random initializations ---
    K = 10  # number of cosine terms
    np.random.seed(42)
    
    for trial in range(15):
        if _time_left() < 10:
            break
        
        # Random initialization
        coeffs = np.zeros(K + 1)
        coeffs[0] = 1.0  # baseline
        coeffs[1:] = np.random.randn(K) * 0.1
        
        # Optimize using scipy
        result = minimize(
            _score_from_coeffs,
            coeffs,
            args=(N,),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.fun < best_score:
            h = _cosine_parameterize(result.x, N)
            h = np.maximum(h, 0)
            score = _autoconv_peak_ratio(h)
            if score < best_score:
                best_score = score
                best_h = h
    
    # --- Phase 2: Projected gradient descent from best cosine result ---
    if _time_left() > 30:
        h_pgd, val_pgd = _projected_gradient_descent(best_h, max_iters=1500, lr_init=0.01)
        if val_pgd < best_score:
            best_score = val_pgd
            best_h = h_pgd
    
    # --- Phase 3: Simulated annealing polish ---
    if _time_left() > 20:
        h_sa, val_sa = _simulated_annealing(best_h, max_iters=100)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa
    
    # --- Phase 4: More gradient descent iterations if time allows ---
    if _time_left() > 25:
        h_pgd2, val_pgd2 = _projected_gradient_descent(best_h, max_iters=1000, lr_init=0.005)
        if val_pgd2 < best_score:
            best_score = val_pgd2
            best_h = h_pgd2
    
    # Safety: never return worse than uniform
    final_score = _autoconv_peak_ratio(best_h)
    if final_score > 1.0 + 1e-9:
        best_h = np.ones(N)
    
    best_h = np.clip(np.asarray(best_h, dtype=float), 0.0, None)
    return [float(v) for v in best_h]

_start_time = 0.0
