"""
Strategy: Golay Complementary Pair Optimization.

The objective is to minimize V = |2 * N * max(conv(h, h)) / sum(h)^2|.
We represent h as the sum of two sequences s1 and s2: h = s1 + s2.
This is inspired by Golay complementary sequences, where the sum of the
individual aperiodic autocorrelations of s1 and s2 is zero for all non-zero
lags. If we can find s1, s2 such that their combined autocorrelation has
very low peaks (ideally only at lag 0), we can significantly reduce the
value.

We parameterize s1 and s2 as real-valued vectors.
To encourage "binary-like" or low-peak behavior, we can use a soft constraint
or simply rely on the optimizer to find low-amplitude, structured solutions.
However, simple random initialization might not find the Golay structure.

We will use a smooth surrogate for the max function:
max(x) ~ alpha * log(sum(exp(x_i / alpha)))
This makes the objective differentiable.

We optimize the coefficients of s1 and s2 directly.
To handle the scale invariance, we can fix the sum of h to be 1 (or N) during
optimization, or just let the optimizer handle it since V is scale invariant.
Actually, V is scale invariant, so we can fix sum(h) = 1 to remove the gauge
freedom and improve numerical stability.
Let h = (s1 + s2) / sum(s1 + s2).
Then V = 2 * N * max(conv(h, h)).
Minimizing max(conv(h, h)) is equivalent to minimizing V.

We will use L-BFGS-B to minimize the surrogate objective.
We will perform multiple restarts with different random initializations.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 140.0

def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_max(x, alpha=0.1):
    """Smooth approximation of max(x)."""
    # Shift for numerical stability
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return x_max
    # Use log-sum-exp trick
    # log(sum(exp(x/alpha))) * alpha
    # To avoid overflow, subtract max
    x_shifted = (x - x_max) / alpha
    # Clip to prevent overflow in exp
    x_shifted = np.clip(x_shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(x_shifted)))
    return x_max + alpha * log_sum_exp

def _objective(params, alpha=0.1):
    """
    Objective function for the optimizer.
    params: vector of length 2*N. First N are s1, next N are s2.
    """
    s1 = params[:N]
    s2 = params[N:]
    h_raw = s1 + s2
    
    s_sum = np.sum(h_raw)
    if abs(s_sum) < 1e-6:
        return 1e10
    
    # Normalize h so that sum(h) = 1. This fixes the scale.
    h = h_raw / s_sum
    
    # Compute autoconvolution
    conv = np.convolve(h, h)
    
    # We want to minimize the maximum peak.
    # The lag 0 term is sum(h^2). For sum(h)=1, min sum(h^2) is 1/N (uniform).
    # max(conv) is at least 1/N.
    # V = 2 * N * max(conv).
    # So we minimize max(conv).
    
    # Use surrogate max
    # We only care about the peaks, so we can just use the surrogate on the whole conv.
    # However, conv has 2N-1 elements.
    # To speed up, we can subsample or just use all.
    # N=400, conv length 799. This is small.
    
    obj = _surrogate_max(conv, alpha)
    
    # Add a small penalty for large values of s1, s2 to keep them bounded?
    # Not strictly necessary if we normalize, but helps conditioning.
    # The scale of s1, s2 is free. Let's add a small L2 penalty on s1, s2
    # to prevent them from drifting to infinity while maintaining ratio.
    # Actually, since h = h_raw / sum(h_raw), the objective is invariant to scaling
    # of (s1, s2). This creates a flat direction.
    # To fix this, we can add a penalty ||s1||^2 + ||s2||^2.
    penalty = 1e-6 * (np.sum(s1**2) + np.sum(s2**2))
    
    return obj + penalty

def construct():
    """
    Construct step heights using Golay-like pair optimization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = _evaluate(np.ones(N))
    best_h = np.ones(N).copy()
    
    # Number of restarts
    n_restarts = 30
    
    rng = np.random.default_rng(42)
    
    # Alpha for the soft-max. Smaller alpha -> sharper max, harder to optimize.
    # Larger alpha -> smoother, easier to optimize, but less accurate.
    # Start with a larger alpha and anneal down?
    # Or just use a fixed moderate alpha.
    alpha_start = 0.5
    alpha_end = 0.05
    
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Interpolate alpha
        alpha = alpha_start + (alpha_end - alpha_start) * (i / n_restarts)
        
        # Initialize s1, s2
        # Try different initializations:
        # 1. Random noise
        # 2. Low frequency smooth
        # 3. Binary-like (random signs)
        
        init_type = i % 3
        
        if init_type == 0:
            # Random noise
            s1 = rng.normal(0, 1, N)
            s2 = rng.normal(0, 1, N)
        elif init_type == 1:
            # Low frequency
            t = np.arange(N)
            s1 = np.sin(2 * np.pi * rng.integers(1, 5) * t / N) + 0.5 * np.cos(2 * np.pi * rng.integers(1, 5) * t / N)
            s2 = np.sin(2 * np.pi * rng.integers(1, 5) * t / N + 1.0) + 0.5 * np.cos(2 * np.pi * rng.integers(1, 5) * t / N + 0.5)
        else:
            # Binary-like (random +/- 1)
            s1 = rng.choice([-1, 1], N)
            s2 = rng.choice([-1, 1], N)
        
        params = np.concatenate([s1, s2])
        
        # We need to wrap the objective to use the current alpha
        def obj_wrapper(p):
            return _objective(p, alpha)
        
        try:
            res = minimize(
                obj_wrapper,
                params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < 1e10:
                h_opt_raw = res.x[:N] + res.x[N:]
                s_sum = np.sum(h_opt_raw)
                if abs(s_sum) > 1e-6:
                    h_opt = h_opt_raw / s_sum
                    v_opt = _evaluate(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            pass

    # Final polish: Greedy refinement on the full vector
    # This helps fix any discrepancies between the surrogate and the true max
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.05
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.95
            else:
                step_size = min(step_size * 1.1, 0.2)
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
