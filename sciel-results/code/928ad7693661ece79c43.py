import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_DEADLINE = time.time() + 110.0  # Safety margin

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return float("inf")
    # Use FFT for fast convolution
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, beta=50.0):
    """Smooth approximation of max using log-sum-exp for gradient optimization."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return float("inf")
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    m = np.max(conv)
    # Log-sum-exp trick
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

# DCT basis for parameterization to ensure real-valued outputs
_K_DCT = 30
_n_idx = np.arange(N)
_k_idx = np.arange(_K_DCT)
_angles = np.pi * (2 * _n_idx[:, None] + 1) * _k_idx[None, :] / (2 * N)
_DCT_BASIS = np.cos(_angles)
_norms = np.sqrt(np.sum(_DCT_BASIS ** 2, axis=0))
_norms[_norms == 0] = 1.0
_DCT_BASIS = _DCT_BASIS / _norms

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=50.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def construct():
    """
    Constructs a sequence of 400 real step heights to minimize the autocorrelation inequality value.
    Uses DCT basis parameterization and smooth optimization to find low-peak autoconvolution sequences.
    """
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return time.time() < _DEADLINE

    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v

    # Phase 1: Global search using Differential Evolution on DCT coefficients
    if deadline_ok():
        try:
            bounds = [(-5.0, 5.0)] * _K_DCT
            # Use a subset of DCT coefficients to reduce dimensionality for DE
            res = differential_evolution(
                _dct_obj, bounds, seed=42, maxiter=30, tol=1e-6,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 2: Local refinement using L-BFGS-B on smooth surrogate
    if deadline_ok():
        try:
            # Initialize from the best found so far, projected to DCT coefficients
            # Simple projection: coeffs = DCT_BASIS.T @ h
            coeffs_init = _DCT_BASIS.T @ best_h
            
            res = minimize(
                _dct_smooth_obj, coeffs_init, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 3: Direct local refinement on h itself if time permits
    if deadline_ok():
        try:
            res = minimize(
                lambda x: _smooth_max_value(x, beta=100.0), best_h, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.x is not None:
                h = res.x
                update_best(h)
        except Exception:
            pass

    return best_h
