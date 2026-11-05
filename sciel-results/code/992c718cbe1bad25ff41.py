"""Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu +
full-dimensional L-BFGS-B with analytical gradients + smooth-max refinement.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate with gradients.
- Secondary: Golay complementary pair (two chirps with different rates).
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional L-BFGS-B with analytical gradient of smooth max.
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
_K_DCT = 80  # Increased DCT modes for more flexibility
_K_DFT = 40  # Increased DFT modes for refinement

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

def _smooth_max_value_and_grad(h, beta=150.0):
    """
    Smooth approximation of max of autoconvolution using log-sum-exp.
    Returns (value, gradient) where gradient is d(value)/d(h).
    
    The autoconvolution at lag k is sum_i h[i]*h[i+k].
    d(conv_k)/d(h[j]) = h[j-k] + h[j+k] (with boundary handling)
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    log_sum_exp = m + np.log(np.sum(exp_scaled)) / beta
    
    # Gradient of log-sum-exp with respect to conv
    weights = exp_scaled / np.sum(exp_scaled)  # softmax weights
    
    # Gradient of value = 2*N*log_sum_exp / s^2 with respect to h
    # d(value)/d(h[j]) = (2*N/s^2) * sum_k weights[k] * d(conv_k)/d(h[j])
    # d(conv_k)/d(h[j]) = h[j-k] + h[j+k] for valid indices
    
    grad_h = np.zeros_like(h)
    for k in range(len(conv)):
        if weights[k] < 1e-10:
            continue
        # conv[k] = sum_i h[i]*h[i+k] for i=0..N-1-k
        # d(conv[k])/d(h[j]) = h[j-k] (if j>=k) + h[j+k] (if j+k<N)
        w = weights[k]
        # Contribution from h[j-k]*h[j] term: for i=j-k, need j-k >= 0 and j-k+k = j < N
        if k == 0:
            # conv[0] = sum h[i]^2, d/dh[j] = 2*h[j]
            grad_h[j] += w * 2.0 * h[j] if False else 0  # placeholder
        else:
            # For lag k > 0:
            # d(conv[k])/d(h[j]) = h[j-k] (if 0 <= j-k < N-k) + h[j+k] (if j+k < N)
            # Actually conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
            # d/dh[j]: if j < N-k, term h[j]*h[j+k] contributes h[j+k]
            #           if j >= k, term h[j-k]*h[j] contributes h[j-k]
            if j < N - k:
                pass  # will handle in vectorized way below
            if j >= k:
                pass
    
    # Vectorized gradient computation
    # For each lag k, d(conv[k])/d(h) is a vector where:
    # (d(conv[k])/d(h))[j] = h[j-k] (if j>=k and j-k < N-k) + h[j+k] (if j+k < N)
    # Simplify: for lag k, the gradient vector g_k has:
    # g_k[j] = h[j-k] for j in [k, N-1] (from the h[j-k]*h[j] term)
    #          + h[j+k] for j in [0, N-1-k] (from the h[j]*h[j+k] term)
    
    grad_h = np.zeros_like(h)
    for k in range(len(conv)):
        if weights[k] < 1e-12:
            continue
        w = weights[k]
        if k == 0:
            grad_h += w * 2.0 * h
        else:
            # h[j-k] for j >= k: indices j from k to N-1, h indices 0 to N-1-k
            grad_h[k:] += w * h[:N-k]
            # h[j+k] for j < N-k: indices j from 0 to N-1-k, h indices k to N-1
            grad_h[:N-k] += w * h[k:]
    
    # Now compute gradient of value = 2*N*log_sum_exp / s^2
    # d(value)/d(h[j]) = (2*N/s^2) * grad_h[j] + log_sum_exp * d(2*N/s^2)/d(h[j])
    # d(2*N/s^2)/d(h[j]) = 2*N * (-2)/s^3 = -4*N/s^3
    # So d(value)/d(h[j]) = (2*N/s^2) * grad_h[j] - (4*N/s^3) * log_sum_exp
    
    # But we need abs(), so sign matters. Let's compute without abs first.
    val_noabs = 2.0 * N * log_sum_exp / (s * s)
    if val_noabs >= 0:
        grad = (2.0 * N / (s * s)) * grad_h - (4.0 * N / (s ** 3)) * log_sum_exp
    else:
        grad = -( (2.0 * N / (s * s)) * grad_h - (4.0 * N / (s ** 3)) * log_sum_exp )
    
    return abs(val_noabs), grad

def _smooth_max_value(h, beta=150.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    v, _ = _smooth_max_value_and_grad(h, beta)
    return v

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

def _dct_smooth_obj_and_grad(coeffs, beta=150.0):
    """Smooth objective and gradient for DCT coefficients."""
    h = _dct_h(coeffs)
    val, grad_h = _smooth_max_value_and_grad(h, beta)
    # Chain rule: d(val)/d(coeffs) = DCT_BASIS^T @ grad_h
    grad_coeffs = _DCT_BASIS_T @ grad_h
    return val, grad_coeffs

def _dct_smooth_obj(coeffs, beta=150.0):
    """Smooth objective for DCT coefficients."""
    val, _ = _dct_smooth_obj_and_grad(coeffs, beta)
    return val

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

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

def _dft_smooth_obj(coeffs_base, beta=150.0):
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
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=150.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = _i_arr
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    
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

def _windowed_zc_smooth_obj(params, beta=150.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

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

def _full_dim_smooth_obj_and_grad(h, beta=150.0):
    """Smooth objective and gradient for full-dimensional h."""
    return _smooth_max_value_and_grad(h, beta)

def _full_dim_lbfgsb(h0, max_iter=200, beta=150.0):
    """
    Full-dimensional L-BFGS-B with analytical gradient.
    """
    def obj_and_grad(h):
        val, grad = _full_dim_smooth_obj_and_grad(h, beta)
        if not np.all(np.isfinite(grad)):
            grad = np.zeros_like(h)
        return val, grad
    
    res = minimize(obj_and_grad, h0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12,
                            'maxls': 50})
    if res.x is not None and np.all(np.isfinite(res.x)):
        return res.x, res.fun
    return h0, _smooth_max_value(h0, beta)

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
    
    # Phase 1: DCT spectral shaping with L-BFGS-B and gradients (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(10):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(6, min(20, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(lambda c: _dct_smooth_obj(c, beta=150.0), coeffs,
                              method='L-BFGS-B',
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),       # phase1
                (-50.0, 50.0),         # rate1
                (-3.0, 3.0),           # amp1
                (-np.pi, np.pi),       # phase2
                (-50.0, 50.0),         # rate2
                (-3.0, 3.0),           # amp2
                (-3.0, 3.0),           # base
            ]
            res_golay = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_complementary_h(res_golay.x)
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
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=45, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
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
                _chirp_obj, bounds_chirp, seed=99, maxiter=30, tol=1e-6,
                popsize=8, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: Full-dimensional L-BFGS-B with analytical gradients
    if deadline_ok():
        try:
            h_fd, v_fd = _full_dim_lbfgsb(best_h, max_iter=300, beta=150.0)
            update_best(h_fd)
        except Exception:
            pass
    
    # Phase 6: L-BFGS-B refinement on DFT basis from best solution
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_dft = minimize(
                lambda x: _dft_smooth_obj(x, beta=150.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft.x is not None:
                h = _dft_h(res_dft.x[:-1], res_dft.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: L-BFGS-B refinement on Golay complementary parameters
    if deadline_ok():
        try:
            x0_golay = np.array([0.0, 5.0, 1.0, np.pi/2, -7.0, 0.8, 0.5])
            
            res_golay_refine = minimize(
                lambda x: _golay_smooth_obj(x, beta=150.0),
                x0_golay,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_golay_refine.x is not None:
                h = _golay_complementary_h(res_golay_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
            res_dct_refine = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=150.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dct_refine.x is not None:
                h = _dct_h(res_dct_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 9: Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_nm.x is not None:
                h = _dft_h(res_nm.x[:-1], res_nm.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 10: Final full-dimensional L-BFGS-B polish with higher beta
    if deadline_ok():
        try:
            h_fd2, v_fd2 = _full_dim_lbfgsb(best_h, max_iter=200, beta=200.0)
            update_best(h_fd2)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
