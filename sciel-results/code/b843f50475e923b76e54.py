"""
Strategy: Windowed Zadoff-Chu Sequence with Direct 400-Dim Powell Optimization

Instead of restricting the signal to a simple sinusoidal chirp, we use the real part
of a Zadoff-Chu sequence. Zadoff-Chu sequences have ideal periodic autocorrelation
properties (zero sidelobes for periodic correlation). By taking the real part and
applying a smooth window (e.g., Gaussian or raised cosine), we create a signal that
inherits some of the low-peak structure but is aperiodic.

We then perform a direct optimization on all 400 variables using Powell's method,
seeded by the best result from a coarse grid search over the Zadoff-Chu parameters.
This allows the optimizer to fine-tune individual samples to further reduce the
peak of the autoconvolution, which is the target metric.

Key difference from previous strategies:
1. Uses Zadoff-Chu (complex exponential with quadratic phase) instead of simple cos(chirp).
2. Uses direct 400-dim optimization (Powell) rather than parametric 5-dim optimization.
3. The "structure" is a starting point for a full vector search, not the final constraint.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _generate_zadoff_chu(l, u, N):
    """
    Generate a Zadoff-Chu sequence of length N.
    h[n] = exp(j * pi * u * n * (n + (1 if l odd else 0)) / l)
    We take the real part.
    """
    n = np.arange(N)
    # Ensure l is odd for standard ZC definition, or use general quadratic phase
    # General chirp: exp(j * pi * u * n * (n + 1) / l)
    # Let's use a generic quadratic phase which covers ZC properties
    # Phase = pi * u * n * (n + 1) / l
    phase = np.pi * u * n * (n + 1) / l
    return np.cos(phase)

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Coarse grid search over Zadoff-Chu parameters and window width
    # Parameters: l (denominator), u (numerator), width (Gaussian sigma), bias
    # l: odd integers between 3 and 20
    # u: integers coprime to l, between 1 and l
    # width: controls the Gaussian window. Small width = sparse, large = flat.
    # bias: DC offset.
    
    l_candidates = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    u_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    width_candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    bias_candidates = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    n = np.arange(N)
    center = (N - 1) / 2.0
    
    def _make_h(l, u, width, bias):
        base = _generate_zadoff_chu(l, u, N)
        # Gaussian window
        w = np.exp(-((n - center) ** 2) / (2 * width * width))
        h = bias + w * base
        return h

    # Grid search
    found_best = False
    for l in l_candidates:
        if time.time() > deadline - 30: break
        for u in u_candidates:
            if np.gcd(l, u) != 1: continue
            for width in width_candidates:
                for bias in bias_candidates:
                    h = _make_h(l, u, width, bias)
                    val = _evaluate(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
                        found_best = True
    
    if not found_best:
        # Fallback to seed if grid search didn't improve
        return best_h

    # Local refinement using Powell's method on the full 400-dim vector
    # Powell is good for this as it doesn't require gradients and handles
    # non-smooth objectives (max function) reasonably well.
    
    if time.time() < deadline - 10:
        try:
            # We optimize the vector h directly
            # To help Powell, we can fix the scale or normalize, but the objective
            # is scale-invariant. However, Powell might drift.
            # We just minimize the objective directly.
            
            def _obj(h_vec):
                # Ensure finite
                if not np.all(np.isfinite(h_vec)):
                    return 1e10
                return _evaluate(h_vec)
            
            res = minimize(
                _obj,
                best_h,
                method='Powell',
                options={
                    'maxiter': 200,
                    'xtol': 1e-8,
                    'ftol': 1e-8,
                    'disp': False
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_h = res.x
                
        except Exception as e:
            pass
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
        
    return best_h
