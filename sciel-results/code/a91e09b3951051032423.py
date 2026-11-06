"""Rewritten program: FIR spectral shaping with log-domain coefficient optimization,
combined with DCT spectral shaping, Golay complementary pairs, and full-dimensional
refinement.

Key improvements:
- FIR filter parameterization: h = FIR impulse response approximating a Gaussian,
  optimized in log-domain for smoothness and low peak-to-average ratio.
- Increased DCT dimension for more spectral freedom.
- Better gradient-based refinement with exact piecewise-linear optimization.
- Deadline guard with safety margin.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 138.0  # safety margin under 150s wall-clock
_K_DCT = 60
_K_DFT = 30

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

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

# DFT basis for refinement
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

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
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

def _golay_smooth_obj(params, beta=80.0):
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

def _windowed_zc_smooth_obj(params, beta=80.0):
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

def _full_dim_smooth_obj(h, beta=80.0):
    """Smooth objective for full-dimensional h."""
    return _smooth_max_value(h, beta)

def _full_dim_smooth_grad(h, beta=80.0):
    """
    Gradient of the smooth max objective w.r.t. h.
    Uses the log-sum-exp smoothing of max(conv).
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.zeros_like(h)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp == 0:
        return np.zeros_like(h)
    
    weights = exp_scaled / sum_exp  # shape (2N-1,)
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    W = np.fft.rfft(weights, size)
    
    B = np.fft.irfft(W * H, size)[:N]
    A = np.fft.irfft(np.conj(H) * W, size)[:N]
    
    dL_dh = A + B  # shape (N,)
    
    L = m + np.log(sum_exp) / beta
    
    val = 2.0 * N * L / (s * s)
    sign = 1.0 if val >= 0 else -1.0
    grad = sign * 2.0 * N * (dL_dh * s - 2.0 * L) / (s ** 3)
    
    return grad

def _full_dim_lbfgsb(h0, max_iter=200, beta=80.0):
    """
    Full-dimensional L-BFGS-B optimization with smooth surrogate.
    """
    h = h0.copy()
    
    def obj_and_grad(x):
        v = _smooth_max_value(x, beta)
        g = _full_dim_smooth_grad(x, beta)
        return v, g
    
    res = minimize(obj_and_grad, h, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
    
    if res.x is not None and np.all(np.isfinite(res.x)):
        return res.x
    return h

def _exact_max_grad_refine(h0, max_iter=100):
    """
    Refine using the exact gradient when we know which lag achieves the max.
    The objective is piecewise linear in h when the argmax is fixed.
    Vectorized gradient computation for speed.
    """
    h = h0.copy()
    
    for it in range(max_iter):
        s = float(np.sum(h))
        if s == 0 or not np.all(np.isfinite(h)):
            break
        
        conv = _fft_convolve(h)
        k_max = int(np.argmax(conv))
        
        j_arr = np.arange(N)
        idx1 = k_max - j_arr
        idx2 = j_arr - k_max
        
        valid1 = (idx1 >= 0) & (idx1 < N)
        valid2 = (idx2 >= 0) & (idx2 < N)
        
        h_idx1 = np.where(valid1, h[np.clip(idx1, 0, N-1)], 0.0)
        h_idx2 = np.where(valid2, h[np.clip(idx2, 0, N-1)], 0.0)
        grad_conv = h_idx1 + h_idx2
        
        val = 2.0 * N * conv[k_max] / (s * s)
        sign = 1.0 if val >= 0 else -1.0
        
        grad_val = sign * 2.0 * N * (grad_conv * s - 2.0 * conv[k_max]) / (s ** 3)
        
        norm_grad = np.linalg.norm(grad_val)
        if norm_grad < 1e-15:
            break
        
        direction = -grad_val / norm_grad
        best_step = 0.01
        best_v = _value(h)
        
        for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
            h_trial = h + step * direction
            v_trial = _value(h_trial)
            if v_trial < best_v:
                best_step = step
                best_v = v_trial
        
        if best_step == 0.0:
            break
        
        h = h + best_step * direction
    
    return h

def _fir_gaussian_h(params):
    """
    FIR filter parameterization: h is the impulse response of an FIR filter
    designed to approximate a continuous Gaussian function.
    params: [sigma_log, center, amplitude, base, phase]
    sigma_log = log(sigma) to ensure positivity
    The FIR taps are optimized in log-domain for smoothness.
    """
    sigma_log, center, amplitude, base, phase = params
    sigma = np.exp(np.clip(sigma_log, -2.0, 3.0))
    
    i = _i_arr
    x = (i - center) / sigma
    gauss = amplitude * np.exp(-0.5 * x * x) * np.cos(phase * i / N)
    h = base + gauss
    return h

def _fir_gaussian_obj(params):
    """Objective for FIR Gaussian parameters."""
    h = _fir_gaussian_h(params)
    return _value(h)

def _fir_gaussian_smooth_obj(params, beta=80.0):
    """Smooth objective for FIR Gaussian parameters."""
    h = _fir_gaussian_h(params)
    return _smooth_max_value(h, beta)

def _multi_gauss_h(params):
    """
    Multi-Gaussian FIR: sum of multiple Gaussian-shaped components with different
    widths and centers, plus a base. This gives more flexibility than a single Gaussian.
    params: [log_sigma1, center1, amp1, phase1, log_sigma2, center2, amp2, phase2, base]
    """
    log_sigma1, center1, amp1, phase1 = params[0], params[1], params[2], params[3]
    log_sigma2, center2, amp2, phase2 = params[4], params[5], params[6], params[7]
    base = params[8]
    
    i = _i_arr
    sigma1 = np.exp(np.clip(log_sigma1, -2.0, 3.0))
    sigma2 = np.exp(np.clip(log_sigma2, -2.0, 3.0))
    
    x1 = (i - center1) / sigma1
    x2 = (i - center2) / sigma2
    
    g1 = amp1 * np.exp(-0.5 * x1 * x1) * np.cos(phase1 * i / N)
    g2 = amp2 * np.exp(-0.5 * x2 * x2) * np.cos(phase2 * i / N)
    
    h = base + g1 + g2
    return h

def _multi_gauss_obj(params):
    """Objective for multi-Gaussian parameters."""
    h = _multi_gauss_h(params)
    return _value(h)

def _multi_gauss_smooth_obj(params, beta=80.0):
    """Smooth objective for multi-Gaussian parameters."""
    h = _multi_gauss_h(params)
    return _smooth_max_value(h, beta)

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
    
    # Phase 1: FIR Gaussian global search
    if deadline_ok():
        try:
            bounds_fir = [
                (-1.0, 2.5),      # log_sigma
                (0.2, 0.8),       # center
                (-3.0, 3.0),      # amplitude
                (-3.0, 3.0),      # base
                (-10.0, 10.0),    # phase
            ]
            res_fir = differential_evolution(
                _fir_gaussian_obj, bounds_fir, seed=42, maxiter=40, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_fir.x is not None:
                h = _fir_gaussian_h(res_fir.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: Multi-Gaussian global search
    if deadline_ok():
        try:
            bounds_mg = [
                (-1.0, 2.5),      # log_sigma1
                (0.1, 0.5),       # center1
                (-3.0, 3.0),      # amp1
                (-10.0, 10.0),    # phase1
                (-1.0, 2.5),      # log_sigma2
                (0.5, 0.9),       # center2
                (-3.0, 3.0),      # amp2
                (-10.0, 10.0),    # phase2
                (-3.0, 3.0),      # base
            ]
            res_mg = differential_evolution(
                _multi_gauss_obj, bounds_mg, seed=7, maxiter=35, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_mg.x is not None:
                h = _multi_gauss_h(res_mg.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(8):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(15, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 4: Golay complementary pair global search
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
                _golay_obj, bounds_golay, seed=7, maxiter=45, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_complementary_h(res_golay.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.3, 1.5),
                (0.2, 0.8),
                (0.0, 1.5),
            ]
            res_wzc = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=40, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_wzc.x is not None:
                h = _windowed_zadoff_chu_h(res_wzc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: Chirp global search for diversity
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_chirp = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=25, tol=1e-6,
                popsize=8, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Full-dimensional L-BFGS-B with smooth surrogate
    if deadline_ok():
        try:
            h_lbfgs = _full_dim_lbfgsb(best_h, max_iter=150, beta=80.0)
            update_best(h_lbfgs)
        except Exception:
            pass
    
    # Phase 8: L-BFGS-B refinement on DFT basis from best solution
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_dft = minimize(
                lambda x: _dft_smooth_obj(x, beta=80.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft.x is not None:
                h = _dft_h(res_dft.x[:-1], res_dft.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 9: L-BFGS-B refinement on Golay complementary parameters
    if deadline_ok():
        try:
            x0_golay = np.array([0.0, 5.0, 1.0, np.pi/2, -7.0, 0.8, 0.5])
            
            res_golay_refine = minimize(
                lambda x: _golay_smooth_obj(x, beta=100.0),
                x0_golay,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_golay_refine.x is not None:
                h = _golay_complementary_h(res_golay_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 10: FIR Gaussian refinement from best solution
    if deadline_ok():
        try:
            # Initialize FIR params from best_h characteristics
            mean_h = float(np.mean(best_h))
            var_h = float(np.var(best_h))
            std_h = np.sqrt(var_h) if var_h > 0 else 1.0
            center_est = float(np.sum(best_h * _i_arr) / max(float(np.sum(best_h)), 1e-10)) / N
            center_est = np.clip(center_est, 0.1, 0.9)
            
            x0_fir = np.array([np.log(max(std_h, 0.1)), center_est, 1.0, mean_h, 0.0])
            
            res_fir_refine = minimize(
                lambda x: _fir_gaussian_smooth_obj(x, beta=100.0),
                x0_fir,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_fir_refine.x is not None:
                h = _fir_gaussian_h(res_fir_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 11: Multi-Gaussian refinement from best solution
    if deadline_ok():
        try:
            x0_mg = np.array([0.5, 0.3, 1.0, 0.0, 0.5, 0.7, 0.8, np.pi/4, 0.5])
            
            res_mg_refine = minimize(
                lambda x: _multi_gauss_smooth_obj(x, beta=100.0),
                x0_mg,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_mg_refine.x is not None:
                h = _multi_gauss_h(res_mg_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 12: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
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
    
    # Phase 13: Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm = minimize(
                lambda x: _value(_dft_h(x[:-1], x[-1])),
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_nm.x is not None:
                h = _dft_h(res_nm.x[:-1], res_nm.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 14: Exact max-gradient refinement (piecewise linear optimization)
    if deadline_ok():
        try:
            h_exact = _exact_max_grad_refine(best_h, max_iter=100)
            update_best(h_exact)
        except Exception:
            pass
    
    # Phase 15: Full-dimensional L-BFGS-B polish with higher beta
    if deadline_ok():
        try:
            h_lbfgs2 = _full_dim_lbfgsb(best_h, max_iter=100, beta=100.0)
            update_best(h_lbfgs2)
        except Exception:
            pass
    
    # Phase 16: Another exact gradient refinement pass
    if deadline_ok():
        try:
            h_exact2 = _exact_max_grad_refine(best_h, max_iter=80)
            update_best(h_exact2)
        except Exception:
            pass
    
    # Phase 17: Final high-precision exact gradient refinement
    if deadline_ok():
        try:
            h_exact3 = _exact_max_grad_refine(best_h, max_iter=100)
            update_best(h_exact3)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
