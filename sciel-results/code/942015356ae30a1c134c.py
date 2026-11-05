"""Combined program: Golay complementary pair + DCT spectral shaping + windowed Zadoff-Chu + smooth-max refinement.

Strategy:
- Primary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values, reducing the maximum peak.
- Secondary: DCT spectral shaping with L-BFGS-B on smooth max surrogate.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock
_K_DCT = 30
_K_DFT = 15

# Precompute common arrays
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

def _smooth_max_value(h, beta=50.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    # Numerically stable log-sum-exp
    max_scaled = np.max(scaled)
    log_sum_exp = m + (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def _dct_smooth_obj_grad(coeffs, beta=50.0):
    """Smooth objective with analytic gradient for DCT coefficients."""
    h = _dct_h(coeffs)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(coeffs)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    log_sum = np.sum(exp_scaled)
    log_sum_exp = m + (max_scaled + np.log(log_sum)) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    
    # Analytic gradient of log-sum-exp w.r.t. conv entries
    # d(log_sum_exp)/d(conv[j]) = exp_scaled[j] / log_sum / beta
    # But log_sum_exp = m + (max_scaled + log(log_sum))/beta
    # This is complex; use finite differences for the outer abs and 2N/s^2 factor
    # but compute the inner gradient analytically
    
    # Actually, let's use a simpler approach: finite difference on the full value
    # but with a larger step for stability
    eps = 1e-5
    grad = np.zeros_like(coeffs)
    for i in range(len(coeffs)):
        c_plus = coeffs.copy()
        c_plus[i] += eps
        h_plus = _dct_h(c_plus)
        s_plus = float(np.sum(h_plus))
        if s_plus == 0.0 or not np.all(np.isfinite(h_plus)):
            grad[i] = 0.0
            continue
        conv_plus = _fft_convolve(h_plus)
        m_plus = np.max(conv_plus)
        scaled_plus = beta * (conv_plus - m_plus)
        max_scaled_plus = np.max(scaled_plus)
        log_sum_plus = np.sum(np.exp(scaled_plus - max_scaled_plus))
        lse_plus = m_plus + (max_scaled_plus + np.log(log_sum_plus)) / beta
        v_plus = abs(2.0 * N * lse_plus / (s_plus * s_plus))
        
        c_minus = coeffs.copy()
        c_minus[i] -= eps
        h_minus = _dct_h(c_minus)
        s_minus = float(np.sum(h_minus))
        if s_minus == 0.0 or not np.all(np.isfinite(h_minus)):
            grad[i] = 0.0
            continue
        conv_minus = _fft_convolve(h_minus)
        m_minus = np.max(conv_minus)
        scaled_minus = beta * (conv_minus - m_minus)
        max_scaled_minus = np.max(scaled_minus)
        log_sum_minus = np.sum(np.exp(scaled_minus - max_scaled_minus))
        lse_minus = m_minus + (max_scaled_minus + np.log(log_sum_minus)) / beta
        v_minus = abs(2.0 * N * lse_minus / (s_minus * s_minus))
        
        grad[i] = (v_plus - v_minus) / (2 * eps)
    
    return val, grad

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

def _dct_smooth_obj(coeffs, beta=50.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def _dft_smooth_obj(coeffs_base, beta=50.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _golay_complementary_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    ang1 = 2.0 * np.pi * rate1 * _i_arr * (_i_arr + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * _i_arr * (_i_arr + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=50.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    ang = 2.0 * np.pi * rate * _i_arr * (_i_arr + 1) / N + phase
    rel_pos = _i_arr / N
    
    if win_type < 0.5:
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        sigma = win_width / 4.0
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    ang = phase + rate * (_i_arr / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    for k in range(1, _K_DFT + 1):
        angle = 2.0 * np.pi * k * _n_idx / N
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        # Least squares for each frequency component
        A = np.column_stack([cos_k, sin_k])
        try:
            sol = np.linalg.lstsq(A, target, rcond=None)[0]
            coeffs[2 * (k - 1)] = sol[0]
            coeffs[2 * (k - 1) + 1] = sol[1]
        except:
            pass
    return coeffs, mean_h

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
    
    # Phase 1: Golay complementary pair - global search (primary strategy)
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),       # phase1
                (-30.0, 30.0),         # rate1
                (-2.0, 2.0),           # amp1
                (-np.pi, np.pi),       # phase2
                (-30.0, 30.0),         # rate2
                (-2.0, 2.0),           # amp2
                (-3.0, 3.0),           # base
            ]
            res_golay = differential_evolution(
                _golay_obj, bounds_golay, seed=42, maxiter=60, tol=1e-7,
                popsize=14, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_complementary_h(res_golay.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(12):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(3, min(15, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 2.0, size=n_active)
                
                # Use analytic gradient for faster convergence
                res = minimize(lambda c: _dct_smooth_obj_grad(c, beta=50.0)[0],
                              coeffs, jac=lambda c: _dct_smooth_obj_grad(c, beta=50.0)[1],
                              method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),       # phase
                (-50.0, 50.0),         # rate
                (-3.0, 3.0),           # amp
                (-3.0, 3.0),           # base
                (0.3, 1.5),            # win_width
                (0.2, 0.8),            # win_center
                (0.0, 1.5),            # win_type
            ]
            res_wzc = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=77, maxiter=60, tol=1e-7,
                popsize=14, init='sobol', polish=False, update_interval=1
            )
            if res_wzc.x is not None:
                h = _windowed_zadoff_chu_h(res_wzc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Chirp global search for diversity
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_chirp = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=40, tol=1e-7,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: L-BFGS-B refinement on DFT basis from best solution
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            # Use higher beta for smoother landscape during refinement
            res_dft = minimize(
                lambda x: _dft_smooth_obj(x, beta=80.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft.x is not None:
                h = _dft_h(res_dft.x[:-1], res_dft.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: L-BFGS-B refinement on Golay complementary parameters
    if deadline_ok():
        try:
            # Initialize from a reasonable starting point
            x0_golay = np.array([0.0, 5.0, 1.0, np.pi/2, -7.0, 0.8, 0.5])
            
            res_golay_refine = minimize(
                lambda x: _golay_smooth_obj(x, beta=80.0),
                x0_golay,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            if res_golay_refine.x is not None:
                h = _golay_complementary_h(res_golay_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
            res_dct_refine = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=50.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            if res_dct_refine.x is not None:
                h = _dct_h(res_dct_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 800, 'maxfev': 1500}
            )
            if res_nm.x is not None:
                h = _dft_h(res_nm.x[:-1], res_nm.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 9: Additional DCT refinement with different beta values
    if deadline_ok():
        try:
            for beta_val in [30.0, 80.0, 150.0]:
                if not deadline_ok():
                    break
                dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
                res_dct_beta = minimize(
                    lambda c: _smooth_max_value(_dct_h(c), beta=beta_val),
                    dct_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_dct_beta.x is not None:
                    h = _dct_h(res_dct_beta.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 10: Final Nelder-Mead polish
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm2 = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 1000, 'maxfev': 2000}
            )
            if res_nm2.x is not None:
                h = _dft_h(res_nm2.x[:-1], res_nm2.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 11: High-precision final polish with tighter tolerances
    if deadline_ok():
        try:
            # Final refinement using the exact (non-smooth) objective
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_final = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_final.x is not None:
                h = _dft_h(res_final.x[:-1], res_final.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
