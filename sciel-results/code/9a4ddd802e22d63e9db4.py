"""
Third autocorrelation inequality - stronger algorithm
Uses iterative reweighted least squares (IRLS) on autoconvolution peak,
combined with DCT spectral shaping and full-dimensional coordinate descent.
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
_n_idx = np.arange(N)

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

def _smooth_max_value(h, beta=100.0):
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

def _dct_smooth_obj(coeffs, beta=100.0):
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

def _chirp_smooth_obj(params, beta=100.0):
    """Smooth objective for chirp parameters."""
    h = _chirp_h(params)
    return _smooth_max_value(h, beta)

def _golay_h(params):
    """Golay complementary pair with two chirps."""
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

def _golay_smooth_obj(params, beta=100.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zc_h(params):
    """Windowed Zadoff-Chu with smooth window."""
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
    h = _windowed_zc_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=100.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zc_h(params)
    return _smooth_max_value(h, beta)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * 20)
    target = h - mean_h
    for k in range(1, 21):
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

def _irwl_refine(h0, max_iter=50, tol=1e-8):
    """
    Iterative Reweighted Least Squares to minimize max autoconvolution.
    This is equivalent to solving a sequence of weighted least squares problems.
    """
    h = h0.copy()
    best_v = _value(h)
    
    for it in range(max_iter):
        conv = _fft_convolve(h)
        # Find the maximum entry and its index
        idx = np.argmax(conv)
        
        # Create weights that emphasize the peak
        # Use exponential weighting around the peak
        weights = np.exp(-np.abs(np.arange(len(conv)) - idx) / (N / 4))
        weights /= np.sum(weights)
        
        # Solve weighted least squares: minimize ||W * (h * h - target)||^2
        # This is a nonlinear problem, so use linearization
        # Gradient of conv with respect to h: conv[i] = sum_j h[j] * h[i-j]
        # d(conv[i])/d(h[k]) = h[i-k] + h[k] (for circular, approximately)
        
        # Use simple coordinate descent on the peak
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005]:
                h[i] = old_val + delta
                v_trial = _value(h)
                if v_trial < best_v:
                    best_v = v_trial
                    improved = True
                    break
                else:
                    h[i] = old_val
            else:
                h[i] = old_val
        
        if not improved:
            break
    
    return h, best_v

def _full_dim_coord_descent(h0, max_iter=200, step_size=0.01, beta=100.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            old_val = h[i]
            for delta in [step_size, -step_size]:
                h[i] = old_val + delta
                v_trial = _smooth_max_value(h, beta)
                if v_trial < best_v:
                    best_v = v_trial
                    improved = True
                    break
                else:
                    h[i] = old_val
            else:
                h[i] = old_val
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
    
    # Phase 4: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.1, 2.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ]
            res_zc = differential_evolution(
                _windowed_zc_obj, bounds_zc, seed=13, maxiter=40, tol=1e-7,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res_zc.x is not None:
                h = _windowed_zc_h(res_zc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: IRLS refinement
    if deadline_ok():
        try:
            h_irls, v_irls = _irwl_refine(best_h, max_iter=50)
            update_best(h_irls)
        except Exception:
            pass
    
    # Phase 6: Full-dimensional coordinate descent with in-place updates
    if deadline_ok():
        try:
            h_cd, v_cd = _full_dim_coord_descent(best_h, max_iter=200, step_size=0.02, beta=100.0)
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 7: DFT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            coeffs, base_val = _project_dft(best_h)
            coeffs_base = np.concatenate([coeffs, [base_val]])
            
            _K_DFT = 20
            _basis = np.zeros((2 * _K_DFT, N))
            for k in range(1, _K_DFT + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                _basis[2 * (k - 1), :] = np.cos(angle)
                _basis[2 * (k - 1) + 1, :] = np.sin(angle)
            _BASIS_T = _basis.T
            
            def _dft_h(coeffs, base_val):
                h = base_val + _BASIS_T @ coeffs
                return h
            
            def _dft_smooth_obj(coeffs_base, beta=100.0):
                coeffs = coeffs_base[:-1]
                base_val = coeffs_base[-1]
                h = _dft_h(coeffs, base_val)
                return _smooth_max_value(h, beta)
            
            res_dft_refine = minimize(
                _dft_smooth_obj,
                coeffs_base,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft_refine.x is not None:
                c = res_dft_refine.x[:-1]
                b = res_dft_refine.x[-1]
                h = _dft_h(c, b)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Nelder-Mead polish on DCT basis
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
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
    
    # Phase 9: Final full-dimensional coordinate descent polish with finer steps
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _full_dim_coord_descent(best_h, max_iter=150, step_size=0.005, beta=100.0)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
