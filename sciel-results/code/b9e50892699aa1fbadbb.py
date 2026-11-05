"""Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu + refinement.

Strategy:
- Use DCT basis for smooth signal representation (better than DFT for smooth signals).
- Add Golay complementary pair construction (near-perfect complementary autocorrelation).
- Use windowed Zadoff-Chu with chirp modulation.
- Use FFT-based convolution for speed, enabling more optimization iterations.
- Refine using L-BFGS-B with adaptive smooth max surrogate.
- Deadline guard ensures we return before 150s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct, idct, fft, ifft

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock

def _fft_convolve(h):
    """Fast convolution via FFT (zero-padded to avoid circular wrap-around)."""
    n = len(h)
    size = 2 * n
    f = fft(h, n=size)
    conv = ifft(f * f).real[:2 * n - 1]
    return conv

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, alpha=50.0):
    """Smooth surrogate for the max of convolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    # Soft max via log-sum-exp
    scaled = conv * alpha
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / alpha
    return abs(2.0 * N * soft_max / (s * s))

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase + rate * (i/N)^2
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _windowed_zadoff_chu_h(params):
    """Generate h from windowed Zadoff-Chu with chirp modulation.
    
    params: [phase, rate, amp, base, window_shape]
    window_shape: 0=rectangular, 1=Hann, 2=Hamming, 3=Blackman
    """
    phase, rate, amp, base, win_shape = params
    i = np.arange(N, dtype=float)
    # ZC-like quadratic phase centered
    ang = phase + rate * i * (i - N / 2.0) / N
    h_raw = base + amp * np.cos(ang)
    
    # Apply window
    if win_shape < 0.5:
        window = np.ones(N)
    elif win_shape < 1.5:
        window = np.hanning(N)
    elif win_shape < 2.5:
        window = np.hamming(N)
    else:
        window = np.blackman(N)
    
    h = h_raw * window
    # Ensure sum is not zero by adding small constant if needed
    if abs(np.sum(h)) < 1e-10:
        h += 0.01
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _golay_complementary_h(params):
    """Generate h from Golay complementary pair construction.
    
    Golay sequences have the property that the sum of their autocorrelations
    is zero for all non-zero shifts. We construct h as a linear combination
    of two complementary sequences with adjustable parameters.
    
    params: [a, b, phase1, phase2, rate1, rate2, base]
    """
    a, b, phase1, phase2, rate1, rate2, base = params
    i = np.arange(N, dtype=float)
    
    # Two chirp-like sequences with different rates
    ang1 = phase1 + rate1 * (i / N) ** 2
    ang2 = phase2 + rate2 * (i / N) ** 2
    
    # Golay-like combination: use ±1 patterns modulated by chirps
    # For real-valued, use cos with different phases
    h = base + a * np.cos(ang1) + b * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _value(h)

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

# DCT basis for refinement (better for smooth signals)
_K = 20  # number of DCT frequency components
_n_idx = np.arange(N)

# Build DCT basis matrix using scipy's DCT
_dct_basis = np.zeros((_K, N))
for k in range(1, _K + 1):
    # DCT-II basis: cos(pi * k * (n + 0.5) / N)
    _dct_basis[k - 1, :] = np.cos(np.pi * k * (_n_idx + 0.5) / N)

_DCT_BASIS_T = _dct_basis.T  # (N, K)

def _dct_h(coeffs, base_val):
    """Generate h from DCT coefficients and a base value."""
    # coeffs shape: (K,), base_val is scalar
    h = base_val + _DCT_BASIS_T @ coeffs
    return h

def _dct_obj(coeffs_base):
    """Objective for DCT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dct_h(coeffs, base_val)
    return _value(h)

def _dct_smooth_obj(coeffs_base, alpha=50.0):
    """Smooth objective for DCT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dct_h(coeffs, base_val)
    return _smooth_max_value(h, alpha=alpha)

# Keep DFT basis too for variety
_K_dft = 10
_dft_basis = np.zeros((2 * _K_dft, N))
for k in range(1, _K_dft + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _dft_basis[2 * (k - 1), :] = np.cos(angle)
    _dft_basis[2 * (k - 1) + 1, :] = np.sin(angle)
_DFT_BASIS_T = _dft_basis.T  # (N, 2*K_dft)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    # coeffs shape: (2*K_dft,), base_val is scalar
    h = base_val + _DFT_BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def _dft_smooth_obj(coeffs_base, alpha=50.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, alpha=alpha)

def _project_to_dct(h):
    """Project h onto DCT basis, returning (coeffs, base_val)."""
    mean_h = np.mean(h)
    target = h - mean_h
    # Use lstsq to project onto DCT basis
    try:
        coeffs = np.linalg.lstsq(_DCT_BASIS_T, target, rcond=None)[0]
    except:
        coeffs = np.zeros(_K)
    return coeffs, mean_h

def _project_to_dft(h):
    """Project h onto DFT basis, returning (coeffs, base_val)."""
    mean_h = np.mean(h)
    target = h - mean_h
    try:
        coeffs = np.linalg.lstsq(_DFT_BASIS_T, target, rcond=None)[0]
    except:
        coeffs = np.zeros(2 * _K_dft)
    return coeffs, mean_h

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Phase 1: Global search via differential evolution on windowed Zadoff-Chu parameters
    bounds_wzc = [
        (-np.pi, np.pi),      # phase
        (-50.0, 50.0),        # rate
        (-3.0, 3.0),          # amplitude
        (-3.0, 3.0),          # base
        (0.0, 3.5),           # window shape
    ]
    
    if deadline_ok():
        try:
            res = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=42,
                maxiter=40,
                tol=1e-6,
                popsize=12,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res.x is not None:
                h = _windowed_zadoff_chu_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 1b: Global search via differential evolution on Golay complementary parameters
    if deadline_ok():
        try:
            bounds_golay = [
                (-2.0, 2.0),      # a
                (-2.0, 2.0),      # b
                (-np.pi, np.pi),  # phase1
                (-np.pi, np.pi),  # phase2
                (-100.0, 100.0),  # rate1
                (-100.0, 100.0),  # rate2
                (-3.0, 3.0),      # base
            ]
            res_g = differential_evolution(
                _golay_obj,
                bounds_golay,
                seed=123,
                maxiter=35,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_g.x is not None:
                h = _golay_complementary_h(res_g.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on windowed ZC parameters
    if deadline_ok():
        try:
            for start in [
                [0.0, 10.0, 1.0, 1.0, 1.0],
                [0.0, -10.0, 1.0, 1.0, 2.0],
                [np.pi/2, 5.0, 0.5, 1.0, 3.0],
                [0.0, 20.0, 1.5, 0.5, 1.5],
            ]:
                if not deadline_ok():
                    break
                res2 = minimize(
                    _windowed_zc_obj,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 300},
                )
                if res2.x is not None:
                    h = _windowed_zadoff_chu_h(res2.x)
                    v = _value(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement on DCT basis with smooth max surrogate
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_to_dct(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res3 = minimize(
                lambda x: _dct_smooth_obj(x, alpha=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res3.x is not None:
                h = _dct_h(res3.x[:-1], res3.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 4: L-BFGS-B refinement on DFT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_to_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res4 = minimize(
                lambda x: _dft_smooth_obj(x, alpha=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res4.x is not None:
                h = _dft_h(res4.x[:-1], res4.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish on DCT with tighter alpha
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_to_dct(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res5 = minimize(
                lambda x: _dct_smooth_obj(x, alpha=200.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res5.x is not None:
                h = _dct_h(res5.x[:-1], res5.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish on DCT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_to_dct(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res6 = minimize(
                _dct_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000},
            )
            if res6.x is not None:
                h = _dct_h(res6.x[:-1], res6.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
