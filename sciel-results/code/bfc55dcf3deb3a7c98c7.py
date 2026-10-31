"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Use water-filling / projected gradient descent to directly optimize
the 600-dim height vector. Key improvements:
  1. Vectorized autoconvolution via FFT (O(N log N) instead of O(N^2))
  2. Water-filling: raise low-autoconv positions, lower high-autoconv positions
  3. Multiple restarts with different initializations
  4. Final polish with tight tolerance
"""

import time
import math

import numpy as np
from scipy.optimize import differential_evolution

N = 600
DELTA = 0.5 / N
DEADLINE = 140.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_fft(h):
    """Compute autoconvolution of step heights using FFT. O(N log N)."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
    # Scale by delta for actual convolution value
    return C * DELTA


def _evaluate(h):
    """Compute the normalized max autoconvolution score (lower is better).
    
    We minimize: max(autoconv) / (sum(h) * delta)^2
    For uniform h=1: max(autoconv) = N * 1^2 * delta = 0.5
    sum(h)*delta = 0.5
    ratio = 0.5 / 0.25 = 2.0
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    c = _autoconv_fft(h)
    peak = float(np.max(c))
    
    # Normalization: peak / (integral)^2 where integral = sum(h)*delta
    integral = s * DELTA
    return peak / (integral * integral)


def _water_filling_step(h, c, step_size):
    """One water-filling iteration using subgradient of max autoconv.
    
    For the peak lag k*, d(autoconv_{k*})/d(h_i) = (h_{i+k*} + h_{i-k*}) * delta
    """
    k_star = int(np.argmax(c))
    
    # Compute subgradient
    grad = np.zeros(N)
    
    if k_star == 0:
        grad = 2.0 * h * DELTA
    else:
        if k_star < N:
            grad[:N-k_star] += h[k_star:] * DELTA
        if k_star < N:
            grad[k_star:] += h[:N-k_star] * DELTA
    
    # Projected gradient descent with projection to h >= 0 and fixed sum
    target_sum = float(np.sum(h))
    h_new = h - step_size * grad
    h_new = np.clip(h_new, 0.0, None)
    
    # Rescale to maintain sum
    s_new = float(np.sum(h_new))
    if s_new > 1e-12:
        h_new = h_new * (target_sum / s_new)
    
    return h_new


def construct():
    global _t0
    _t0 = time.time()
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _evaluate(best_h)
    
    x_mid = _midpoints()
    
    # Phase 1: Try various parametric families quickly
    for freq in [1, 2, 3, 4]:
        for amp in np.linspace(0.0, 0.5, 11):
            if _time_left() < 10.0:
                break
            h = 1.0 + amp * np.cos(2 * np.pi * freq * x_mid / 0.5)
            h = np.clip(h, 0.0, None)
            score = _evaluate(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Phase 2: Water-filling optimization
    if _time_left() > 30.0:
        h = best_h.copy()
        c = _autoconv_fft(h)
        
        step = 0.01
        for it in range(300):
            if _time_left() < 15.0:
                break
            
            c = _autoconv_fft(h)
            h_new = _water_filling_step(h, c, step)
            score_new = _evaluate(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                step = max(step * 0.995, 0.001)
            else:
                step = max(step * 0.9, 0.001)
        
        # Phase 3: Fine-tune with smaller steps
        if _time_left() > 15.0:
            h = best_h.copy()
            c = _autoconv_fft(h)
            step = 0.001
            for it in range(200):
                if _time_left() < 10.0:
                    break
                
                c = _autoconv_fft(h)
                h_new = _water_filling_step(h, c, step)
                score_new = _evaluate(h_new)
                
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
                    step = max(step * 0.99, 0.0001)
                else:
                    step = max(step * 0.9, 0.0001)
    
    # Phase 4: Fourier parametrization with DE
    if _time_left() > 25.0:
        K = 5
        def fourier_to_heights(params):
            base = params[0]
            h = np.full(N, base)
            for k in range(1, K+1):
                a = params[2*k-1]
                b = params[2*k]
                h += a * np.cos(2*np.pi*k*x_mid/0.5) + b * np.sin(2*np.pi*k*x_mid/0.5)
            return np.clip(h, 0.0, None)
        
        bounds = [(0.5, 2.0)] + [(-0.5, 0.5)] * (2*K)
        
        def objective(params):
            if _time_left() < 5.0:
                return float('inf')
            h = fourier_to_heights(params)
            return _evaluate(h)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=12,
                tol=0.01,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                h_candidate = fourier_to_heights(result.x)
                score = _evaluate(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
