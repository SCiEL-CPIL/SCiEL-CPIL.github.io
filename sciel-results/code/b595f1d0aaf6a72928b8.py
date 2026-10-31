"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Structural change to Spectral Optimization.
  1. Represent the step function by its first K Fourier coefficients.
  2. Use Differential Evolution to find a global optimum in the low-dim spectral space.
  3. Refine using local gradient-based methods in the spectral space.
  4. This avoids the high-dimensional local minima of direct height optimization.
"""

import time
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

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
    return C * DELTA


def _evaluate(h):
    """Compute the normalized max autoconvolution score (lower is better).
    
    We minimize: max(autoconv) / (sum(h) * delta)^2
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    c = _autoconv_fft(h)
    peak = float(np.max(c))
    
    integral = s * DELTA
    return peak / (integral * integral)


def _fourier_to_heights(params, x_mid, K):
    """Convert Fourier coefficients to step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    a0 = params[0]
    h = np.full(N, a0)
    for k in range(1, K+1):
        a = params[2*k-1]
        b = params[2*k]
        # Frequency is k * (2 * pi / 0.5) = 4 * pi * k
        # x is in [-0.25, 0.25]
        # term: a * cos(2*pi*k*x/0.5) + b * sin(2*pi*k*x/0.5)
        h += a * np.cos(4.0 * np.pi * k * x_mid) + b * np.sin(4.0 * np.pi * k * x_mid)
    return h


def _spectral_objective(params, x_mid, K):
    """Objective function for spectral optimization.
    Includes a penalty for negative heights to enforce non-negativity.
    """
    h = _fourier_to_heights(params, x_mid, K)
    
    # Penalty for negative values
    neg_mask = h < 0
    if np.any(neg_mask):
        penalty = 100.0 * np.sum(h[neg_mask] ** 2)
    else:
        penalty = 0.0
        
    h_clipped = np.clip(h, 0.0, None)
    score = _evaluate(h_clipped)
    
    return score + penalty


def construct():
    global _t0
    _t0 = time.time()
    
    x_mid = _midpoints()
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _evaluate(best_h)
    
    # Structural Change: Optimize in Fourier Space
    # We use a moderate number of harmonics K to balance expressiveness and search difficulty
    K = 12 
    num_params = 1 + 2*K
    
    # Bounds for Fourier coefficients
    # a0 is the mean height. Let's center it around 1.0
    # Higher frequencies should have smaller amplitudes to ensure smoothness and non-negativity
    bounds = [(0.5, 2.0)]  # a0
    for k in range(1, K+1):
        # Amplitude decay
        amp_bound = 0.5 / k
        bounds.append((-amp_bound, amp_bound))
        bounds.append((-amp_bound, amp_bound))
        
    # Phase 1: Global Search with Differential Evolution
    if _time_left() > 60.0:
        # Use a subset of the population and maxiter to fit in time
        # DE is stochastic, so we fix seed for reproducibility if needed, but random is fine
        result = differential_evolution(
            _spectral_objective,
            bounds,
            args=(x_mid, K),
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            update_interval=1
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            h_candidate = _fourier_to_heights(best_params, x_mid, K)
            h_candidate = np.clip(h_candidate, 0.0, None)
            score_candidate = _evaluate(h_candidate)
            if score_candidate < best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
    else:
        best_params = None

    # Phase 2: Local Refinement using L-BFGS-B in Spectral Space
    # This is a structural change from direct-space gradient descent to spectral-space local search
    if _time_left() > 20.0:
        if best_params is None:
            # If DE didn't run or failed, start from a reasonable guess
            best_params = np.zeros(num_params)
            best_params[0] = 1.0
            
        # Try a few random restarts in spectral space if time permits
        n_restarts = 3 if _time_left() > 40.0 else 1
        
        for r in range(n_restarts):
            if _time_left() < 10.0:
                break
            
            if r == 0:
                start_params = best_params.copy()
            else:
                # Perturb the best params found so far
                start_params = best_params.copy()
                noise = np.random.normal(0, 0.1, num_params)
                noise[0] = 0 # Don't perturb the mean too much
                start_params += noise
                # Re-apply bounds roughly
                for i in range(1, num_params):
                    k = (i+1)//2
                    amp_bound = 0.5 / k
                    start_params[i] = np.clip(start_params[i], -amp_bound, amp_bound)
            
            # Use L-BFGS-B to refine
            res = minimize(
                _spectral_objective,
                start_params,
                args=(x_mid, K),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.fun < best_score:
                h_candidate = _fourier_to_heights(res.x, x_mid, K)
                h_candidate = np.clip(h_candidate, 0.0, None)
                score_candidate = _evaluate(h_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_h = h_candidate.copy()
                    best_params = res.x.copy()

    # Phase 3: Direct Space Polish (Fallback/Refinement)
    # After finding a good spectral shape, we can do a few steps of direct optimization
    # to fix any non-negativity clipping artifacts or fine-tune individual bins.
    if _time_left() > 5.0:
        h = best_h.copy()
        target_sum = float(np.sum(h))
        
        # Simple projected gradient step for a few iterations
        # Using subgradient of max autoconv
        for it in range(20):
            if _time_left() < 2.0:
                break
                
            c = _autoconv_fft(h)
            peak_idx = np.argmax(c)
            
            # Gradient of max c_k w.r.t h_i
            # c_k = sum_j h_j h_{j+k} delta
            # d c_k / d h_i = h_{i-k} delta + h_{i+k} delta (with boundary handling)
            grad = np.zeros(N)
            k = peak_idx
            if k == 0:
                grad += 2.0 * h * DELTA
            else:
                if k < N:
                    grad[:N-k] += h[k:] * DELTA
                    grad[k:] += h[:N-k] * DELTA
            
            # Normalize
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-12:
                grad /= g_norm
            
            # Step size
            step = 0.05
            h_new = h - step * grad
            h_new = np.clip(h_new, 0.0, None)
            
            # Rescale to maintain integral (normalization invariance)
            s_new = np.sum(h_new)
            if s_new > 1e-12:
                h_new *= (target_sum / s_new)
            
            score_new = _evaluate(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                step *= 0.9
            else:
                step *= 0.8
    
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
