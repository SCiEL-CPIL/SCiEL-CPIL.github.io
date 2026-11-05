"""
Third autocorrelation inequality - combined approach.

Key structural idea (from inspiration 1):
- Use chirp-based parameterizations (Golay complementary pair, windowed Zadoff-Chu,
  multi-harmonic Barker-like) as the primary design family, since these naturally
  suppress sidelobes in autocorrelation.
- Use differential evolution for global search over low-dimensional parameter spaces.

Refinement machinery (from current program):
- FFT-based autoconvolution for speed.
- Smooth-max surrogate (log-sum-exp) for differentiable optimization.
- L-BFGS-B with analytical gradient for local refinement.
- DCT basis projection for final polish.
- Deadline guard with safety margin.

The theoretical lower bound is 2N/(2N-1) ≈ 1.0025 for N=400.
Zadoff-Chu/chirp sequences approach this bound.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct, idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=50.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-15 or not np.all(np.isfinite(h)):
        return 1e10
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    return val

def _smooth_max_value_grad(h, beta=50.0):
    """Compute smooth max value and gradient w.r.t. h."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-15 or not np.all(np.isfinite(h)):
        return 1e10, np.zeros_like(h)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    log_sum_exp = m + np.log(sum_exp) / beta
    
    val = abs(2.0 * N * log_sum_exp / (s * s))
    
    # Weights for smooth max
    weights = exp_scaled / sum_exp
    
    # Gradient of log_sum_exp w.r.t. h
    # grad_h[k] = sum_j weights[j] * d(conv[j])/d(h[k])
    # For full convolution: d(conv[j])/d(h[k]) = h[j-k] + h[k-j] (with boundary)
    # Use direct computation (N=400, 799 conv elements -> ~320k ops)
    n = len(h)
    size_conv = 2 * n - 1
    grad_h = np.zeros(n)
    
    for k in range(n):
        # Part 1: sum_j weights[j] * h[j-k] where 0<=j-k<n
        j_lo = max(0, k)
        j_hi = min(size_conv - 1, k + n - 1)
        if j_lo <= j_hi:
            idx = np.arange(j_lo, j_hi + 1)
            h_idx = idx - k
            grad_h[k] += np.sum(weights[j_lo:j_hi+1] * h[h_idx])
        
        # Part 2: sum_j weights[j] * h[k-j] where 0<=k-j<n
        j_lo2 = max(0, k - n + 1)
        j_hi2 = min(size_conv - 1, k)
        if j_lo2 <= j_hi2:
            idx2 = np.arange(j_lo2, j_hi2 + 1)
            h_idx2 = k - idx2
            grad_h[k] += np.sum(weights[j_lo2:j_hi2+1] * h[h_idx2])
    
    # val = abs(2N * LSE / s^2)
    # Assume positive: val = 2N * LSE / s^2
    # d(val)/d(h[k]) = 2N * (grad_h[k] * s - 2 * LSE) / s^3
    LSE = log_sum_exp
    inner = 2.0 * N * LSE / (s * s)
    sign = 1.0 if inner >= 0 else -1.0
    grad_val = sign * 2.0 * N * (grad_h * s - 2.0 * LSE) / (s ** 3)
    
    return val, grad_val

# DCT basis for refinement
_K_DCT = 40
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
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_smooth_obj_grad(coeffs, beta=50.0):
    h = _dct_h(coeffs)
    val, grad_h = _smooth_max_value_grad(h, beta)
    grad_coeffs = _DCT_BASIS.T @ grad_h
    return val, grad_coeffs

def _dct_smooth_obj_and_grad(coeffs, beta=50.0):
    return _dct_smooth_obj_grad(coeffs, beta)

def _project_to_dct(h_init):
    """Project h onto DCT basis."""
    coeffs = _DCT_BASIS.T @ h_init
    return coeffs

# Chirp-based parameterizations (structural idea from inspiration)
def _golay_h(params):
    """Golay complementary pair: h = base + a1*cos(q1) + a2*cos(q2)."""
    p1, r1, a1, p2, r2, a2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * r1 * i * (i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * i * (i + 1) / N + p2
    return base + a1 * np.cos(ang1) + a2 * np.cos(ang2)

def _golay_obj(params):
    h = _golay_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=50.0):
    h = _golay_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zc_h(params):
    """Windowed Zadoff-Chu sequence."""
    phase, rate, amp, base, win_width, win_center = params
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    x = (rel_pos - win_center) / (0.5 * win_width)
    window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    return base + amp * window * np.cos(ang)

def _windowed_zc_obj(params):
    h = _windowed_zc_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=50.0):
    h = _windowed_zc_h(params)
    return _smooth_max_value(h, beta)

def _barker_like_h(params):
    """Multi-harmonic Barker-like sequence."""
    base = params[0]
    n_harmonics = 8
    amplitudes = params[1:1+n_harmonics]
    phases = params[1+n_harmonics:1+2*n_harmonics]
    i = np.arange(N, dtype=float)
    h = np.full(N, base)
    for k in range(1, n_harmonics + 1):
        h += amplitudes[k-1] * np.cos(2.0 * np.pi * k * i / N + phases[k-1])
    return h

def _barker_like_obj(params):
    h = _barker_like_h(params)
    return _value(h)

def _barker_like_smooth_obj(params, beta=50.0):
    h = _barker_like_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Single chirp: h = base + amp * cos(phase + rate * (i/N)^2)."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * (i / N) ** 2
    return base + amp * np.cos(ang)

def _chirp_obj(params):
    h = _chirp_h(params)
    return _value(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: Global search with differential evolution on chirp parameterizations
    # These are low-dimensional (4-17 params) and naturally suppress sidelobes
    
    # 1a: Golay complementary pair
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),   # p1
                (-50.0, 50.0),    # r1
                (-3.0, 3.0),      # a1
                (-np.pi, np.pi),  # p2
                (-50.0, 50.0),    # r2
                (-3.0, 3.0),      # a2
                (-3.0, 3.0),      # base
            ]
            res = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=40, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # 1b: Windowed Zadoff-Chu
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),   # phase
                (-50.0, 50.0),    # rate
                (-3.0, 3.0),      # amp
                (-3.0, 3.0),      # base
                (0.3, 1.5),       # win_width
                (0.2, 0.8),       # win_center
            ]
            res = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=40, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _windowed_zc_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # 1c: Barker-like multi-harmonic
    if deadline_ok():
        try:
            n_harm = 8
            bounds_barker = [(-2.0, 2.0)]  # base
            for _ in range(n_harm):
                bounds_barker.append((-2.0, 2.0))  # amplitude
            for _ in range(n_harm):
                bounds_barker.append((-np.pi, np.pi))  # phase
            
            res = differential_evolution(
                _barker_like_obj, bounds_barker, seed=23, maxiter=30, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _barker_like_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # 1d: Single chirp
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),   # phase
                (-100.0, 100.0),  # rate
                (-3.0, 3.0),      # amp
                (-3.0, 3.0),      # base
            ]
            res = differential_evolution(
                _chirp_obj, bounds_chirp, seed=31, maxiter=40, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement on smooth surrogate from best solution
    if deadline_ok():
        try:
            coeffs = _project_to_dct(best_h)
            res = minimize(
                _dct_smooth_obj_and_grad,
                coeffs,
                args=(30.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 400},
            )
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B on Golay smooth objective from best Golay-like solution
    if deadline_ok():
        try:
            # Use the best h to initialize Golay parameters
            # Simple heuristic: fit two chirps
            i = np.arange(N, dtype=float)
            # Try a few initializations
            inits_golay = []
            for r1 in [5, 10, 20, -5, -10]:
                for r2 in [5, 10, 20, -5, -10]:
                    if r1 == r2:
                        continue
                    p1 = 0.0
                    p2 = np.pi / 4
                    a1 = 0.5
                    a2 = 0.3
                    base = float(np.mean(best_h))
                    inits_golay.append([p1, r1, a1, p2, r2, a2, base])
            
            best_golay_v = best_v
            best_golay_params = None
            for init_p in inits_golay[:6]:
                if not deadline_ok():
                    break
                res = minimize(
                    _golay_smooth_obj,
                    init_p,
                    args=(30.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 200},
                )
                if res.x is not None:
                    h = _golay_h(res.x)
                    v = _value(h)
                    if v < best_golay_v:
                        best_golay_v = v
                        best_golay_params = res.x.copy()
            
            if best_golay_params is not None:
                h = _golay_h(best_golay_params)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: DCT refinement with multiple restarts
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(4):
                if not deadline_ok():
                    break
                # Start from best solution with small perturbation
                coeffs = _project_to_dct(best_h)
                coeffs += rng.normal(0, 0.1, size=_K_DCT)
                
                res = minimize(
                    _dct_smooth_obj_and_grad,
                    coeffs,
                    args=(20.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 300},
                )
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish in DCT space
    if deadline_ok():
        try:
            coeffs = _project_to_dct(best_h)
            res = minimize(
                _dct_smooth_obj,
                coeffs,
                args=(10.0,),
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 300, 'maxfev': 600},
            )
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: Try a few more global searches with different seeds if time permits
    if deadline_ok():
        try:
            # Another Golay search with different seed
            bounds_golay2 = [
                (-np.pi, np.pi), (-50.0, 50.0), (-3.0, 3.0),
                (-np.pi, np.pi), (-50.0, 50.0), (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _golay_obj, bounds_golay2, seed=99, maxiter=30, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
