"""Combined program: Dual Zadoff-Chu complementary pair + DCT spectral shaping + coordinate descent refinement.

Strategy:
- Primary: Two orthogonal Zadoff-Chu sequences with independent phases/amplitudes.
  The sum of their autocorrelations can cancel sidelobes, approaching the theoretical
  lower bound of ~1.002.
- Secondary: DCT spectral shaping with L-BFGS-B on smooth max surrogate.
- Tertiary: Coordinate descent on the full 400-dim vector with smooth surrogate.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 145s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock
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
    max_scaled = np.max(scaled)
    log_sum_exp = m + (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
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

def _dual_zadoff_chu_h(params):
    """
    Dual Zadoff-Chu complementary pair:
    h = base + amp1*cos(phase1 + rate1*i*(i+1)/N) + amp2*cos(phase2 + rate2*i*(i+1)/N)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    ang1 = phase1 + 2.0 * np.pi * rate1 * _i_arr * (_i_arr + 1) / N
    ang2 = phase2 + 2.0 * np.pi * rate2 * _i_arr * (_i_arr + 1) / N
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _dual_zc_obj(params):
    """Objective for dual Zadoff-Chu parameters."""
    h = _dual_zadoff_chu_h(params)
    return _value(h)

def _dual_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for dual Zadoff-Chu parameters."""
    h = _dual_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    for k in range(1, _K_DFT + 1):
        angle = 2.0 * np.pi * k * _n_idx / N
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        A = np.column_stack([cos_k, sin_k])
        try:
            sol = np.linalg.lstsq(A, target, rcond=None)[0]
            coeffs[2 * (k - 1)] = sol[0]
            coeffs[2 * (k - 1) + 1] = sol[1]
        except:
            pass
    return coeffs, mean_h

def _coordinate_update(h, i, tau=0.5, max_iter=30):
    """
    Find optimal value for h[i] given all other coordinates fixed.
    We minimize the surrogate over h[i].
    """
    h_fixed = h.copy()
    h_old = h[i]
    
    def obj(delta):
        h_new = h_fixed.copy()
        h_new[i] = h_old + delta
        s = np.sum(h_new)
        if abs(s) < 1e-6:
            return 1e10
        h_norm = h_new * N / s
        conv = _fft_convolve(h_norm)
        m = np.max(conv)
        scaled = tau * (conv - m)
        max_scaled = np.max(scaled)
        log_sum_exp = m + (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / tau
        return abs(2.0 * N * log_sum_exp / (s * s))
    
    try:
        res = minimize_scalar(obj, bounds=(-1.0, 1.0), method='bounded',
                              options={'maxiter': max_iter})
        if res.success:
            return h_old + res.x
    except Exception:
        pass
    return h_old

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
    
    # Phase 1: Dual Zadoff-Chu complementary pair - global search (primary strategy)
    if deadline_ok():
        try:
            bounds_dzc = [
                (-np.pi, np.pi),       # phase1
                (-20.0, 20.0),         # rate1
                (-2.0, 2.0),           # amp1
                (-np.pi, np.pi),       # phase2
                (-20.0, 20.0),         # rate2
                (-2.0, 2.0),           # amp2
                (-3.0, 3.0),           # base
            ]
            res_dzc = differential_evolution(
                _dual_zc_obj, bounds_dzc, seed=42, maxiter=80, tol=1e-8,
                popsize=16, init='sobol', polish=False, update_interval=1
            )
            if res_dzc.x is not None:
                h = _dual_zadoff_chu_h(res_dzc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(6):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(3, min(12, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.5, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 3: Coordinate descent on full 400-dim vector with smooth surrogate
    if deadline_ok():
        try:
            h_curr = best_h.copy()
            s = np.sum(h_curr)
            if abs(s) > 1e-6:
                h_curr = h_curr * N / s
            
            taus = [0.5, 0.2, 0.1]
            for tau in taus:
                if not deadline_ok():
                    break
                
                n_passes = 3
                for p in range(n_passes):
                    if not deadline_ok():
                        break
                    
                    order = np.random.default_rng(p * 100 + 1).permutation(N)
                    for idx in order:
                        if not deadline_ok():
                            break
                        new_val = _coordinate_update(h_curr, idx, tau=tau, max_iter=20)
                        h_curr[idx] = new_val
                    
                    v_curr = _value(h_curr)
                    if v_curr < best_v:
                        best_v = v_curr
                        best_h = h_curr.copy()
        except Exception:
            pass
    
    # Phase 4: L-BFGS-B refinement on DFT basis from best solution
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_dft = minimize(
                lambda x: _dft_smooth_obj(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            if res_dft.x is not None:
                h = _dft_h(res_dft.x[:-1], res_dft.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: L-BFGS-B refinement on dual Zadoff-Chu parameters
    if deadline_ok():
        try:
            x0_dzc = np.array([0.0, 5.0, 1.0, np.pi/2, -7.0, 0.8, 0.5])
            
            res_dzc_refine = minimize(
                lambda x: _dual_zc_smooth_obj(x, beta=80.0),
                x0_dzc,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            if res_dzc_refine.x is not None:
                h = _dual_zadoff_chu_h(res_dzc_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 600, 'maxfev': 1000}
            )
            if res_nm.x is not None:
                h = _dft_h(res_nm.x[:-1], res_nm.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Final coordinate descent polish
    if deadline_ok():
        try:
            h_curr = best_h.copy()
            tau_final = 0.05
            rng_final = np.random.default_rng(999)
            
            for _ in range(100):
                if not deadline_ok():
                    break
                idx = rng_final.integers(0, N)
                new_val = _coordinate_update(h_curr, idx, tau=tau_final, max_iter=15)
                h_curr[idx] = new_val
                v_curr = _value(h_curr)
                if v_curr < best_v:
                    best_v = v_curr
                    best_h = h_curr.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
