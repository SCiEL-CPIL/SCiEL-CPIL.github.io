"""
Optimized program for third autocorrelation inequality.
Strategy: DCT spectral shaping with smooth-max surrogate, multiple restarts,
coordinate descent, and Nelder-Mead polish.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0
_K_DCT = 40

_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)
_DCT_BASIS_T = _DCT_BASIS.T

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = _i_arr
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _chirp_smooth_obj(params, beta=80.0):
    """Smooth objective for chirp parameters."""
    h = _chirp_h(params)
    return _smooth_max_value(h, beta)

def _golay_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=80.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_h(params)
    return _smooth_max_value(h, beta)

def _project_dct(h):
    """Project h onto DCT basis, return coefficients."""
    return scipy_dct(h, type=2, n=_K_DCT, norm='ortho')

def _full_dim_coord_descent(h0, max_iter=500, step_size=0.01, beta=80.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            for delta in [step_size, -step_size]:
                h_trial = h.copy()
                h_trial[i] += delta
                v_trial = _smooth_max_value(h_trial, beta)
                if v_trial < best_v:
                    h = h_trial
                    best_v = v_trial
                    improved = True
                    break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return h, best_v

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(10):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(12, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Chirp global search
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_chirp = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=30, tol=1e-6,
                popsize=8, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_golay = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_h(res_golay.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Full-dimensional coordinate descent
    if deadline_ok():
        try:
            h_cd, v_cd = _full_dim_coord_descent(best_h, max_iter=200, step_size=0.02, beta=80.0)
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 5: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            dct_coeffs = _project_dct(best_h)
            
            res_dct_refine = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=80.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dct_refine.x is not None:
                h = _dct_h(res_dct_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish on DCT basis
    if deadline_ok():
        try:
            dct_coeffs = _project_dct(best_h)
            
            res_nm = minimize(
                _dct_obj,
                dct_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_nm.x is not None:
                h = _dct_h(res_nm.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Final full-dimensional coordinate descent polish
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _full_dim_coord_descent(best_h, max_iter=100, step_size=0.005, beta=100.0)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
