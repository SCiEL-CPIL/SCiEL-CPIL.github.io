"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    if n == 0:
        return np.array([])
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    F = np.fft.rfft(x, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]


def _score_from_signal(x):
    """
    Compute the metric: max(|autoconv|) / (sum(x))^2
    Assumption: Excludes zero-lag (index N-1) based on score > 1.0 analysis.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    # Exclude the center peak (zero lag)
    if len(conv) > 0:
        center_idx = N - 1
        # Create a copy to avoid modifying original, or use mask
        # Mask out the center
        mask = np.ones(len(conv), dtype=bool)
        mask[center_idx] = False
        max_abs = np.max(np.abs(conv[mask]))
    else:
        max_abs = 0.0
    return max_abs / (integral * integral)


def _objective_smooth(x):
    """
    Smooth objective for optimization.
    Minimizes the L2 norm of the sidelobes (excluding center) normalized by integral squared.
    This is differentiable and guides the optimizer away from high sidelobes.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    center_idx = N - 1
    mask = np.ones(len(conv), dtype=bool)
    mask[center_idx] = False
    sidelobes = conv[mask]
    # Use L4 norm to penalize peaks more, or L2 for stability. L2 is smoother.
    # Let's use L2 squared (sum of squares)
    val = np.sum(sidelobes ** 2)
    return val / (integral ** 4) # Scale invariant form: (sum c^2) / (sum x)^4 ? 
    # Wait. Metric is max |c| / (sum x)^2.
    # Minimizing sum c^2 is a proxy.
    # To be scale invariant in the proxy:
    # If x -> ax, c -> a^2 c. sum c^2 -> a^4 sum c^2. (sum x)^4 -> a^4 (sum x)^4.
    # So ratio is invariant.


def construct():
    deadline = time.time() + 140
    
    # Initial best: all ones
    best_x = np.ones(N)
    best_score = _score_from_signal(best_x)
    
    # Optimization using L-BFGS-B
    # We optimize the time-domain signal directly.
    # Constraint: sum(x) = 1 is not strictly necessary due to scale invariance, 
    # but helps numerical stability. We can just let it optimize.
    
    def _obj_wrapper(params):
        return _objective_smooth(params)
    
    # Start from ones
    x0 = np.ones(N)
    
    try:
        # L-BFGS-B is good for large scale
        res = minimize(
            _obj_wrapper, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success and res.fun < _objective_smooth(best_x):
            x_opt = res.x
            score_opt = _score_from_signal(x_opt)
            if score_opt < best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Random initial signal
            x0 = rng.normal(0, 1, N)
            # Ensure non-zero sum
            if abs(np.sum(x0)) < 1e-6:
                x0 += 1.0
            
            try:
                res = minimize(
                    _obj_wrapper, 
                    x0, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success:
                    x_opt = res.x
                    score_opt = _score_from_signal(x_opt)
                    if score_opt < best_score:
                        best_x = x_opt
                        best_score = score_opt
            except Exception:
                pass
    
    return best_x
# EVOLVE-BLOCK-END
