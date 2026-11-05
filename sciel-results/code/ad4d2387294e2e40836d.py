import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_K_DCT = 30
_K_DFT = 15

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
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

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = np.arange(N, dtype=float)
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

def _zadoff_chu_h(params):
    """Zadoff-Chu inspired sequence."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_obj(params):
    """Objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _golay_complementary_h(params):
    """
    Golay complementary pair approach.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    The idea is that two chirps with different rates can have complementary
    autocorrelation properties, reducing the peak of the sum.
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
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

def _coordinate_descent_polish(h, max_iter=200, tol=1e-10):
    """
    Coordinate-wise descent on individual h values.
    For each index i, optimize h[i] while keeping others fixed.
    The objective is f(h) = abs(2*N*max(conv(h))/sum(h)^2).
    We use a smooth surrogate and line search.
    """
    h = h.copy()
    current_val = _smooth_max_value(h, beta=80.0)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            # Try small perturbations around h[i]
            best_local = h[i]
            best_local_val = current_val
            
            # Adaptive step size
            scale = max(0.01, abs(h[i]) * 0.1)
            for step in [-3*scale, -2*scale, -scale, -scale/2, scale/2, scale, 2*scale, 3*scale]:
                h_test = h.copy()
                h_test[i] = h[i] + step
                v_test = _smooth_max_value(h_test, beta=80.0)
                if v_test < best_local_val - tol:
                    best_local_val = v_test
                    best_local = h_test[i]
            
            if best_local != h[i]:
                h[i] = best_local
                current_val = best_local_val
                improved = True
        
        if not improved:
            break
    
    return h

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
    
    # Phase 1: DCT spectral shaping with multiple restarts
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(6):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(3, min(10, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Windowed Zadoff-Chu global search
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
            res = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=42, maxiter=40, tol=1e-6,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _windowed_zadoff_chu_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Golay complementary pair search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),
                (-30.0, 30.0),
                (-2.0, 2.0),
                (-np.pi, np.pi),
                (-30.0, 30.0),
                (-2.0, 2.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _golay_obj, bounds_golay, seed=99, maxiter=35, tol=1e-6,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_complementary_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Chirp global search
    if deadline_ok():
        try:
            bounds = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _chirp_obj, bounds, seed=42, maxiter=40, tol=1e-6,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: DFT refinement with smooth max
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K_DFT)
            for k in range(1, _K_DFT + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                A_k = np.column_stack([np.cos(angle), np.sin(angle)])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(_dft_smooth_obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: Nelder-Mead polish on DFT
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K_DFT)
            for k in range(1, _K_DFT + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                A_k = np.column_stack([np.cos(angle), np.sin(angle)])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(_dft_obj, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 300, 'maxfev': 500})
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Coordinate-wise polish on the best solution
    if deadline_ok():
        try:
            h_polished = _coordinate_descent_polish(best_h, max_iter=50, tol=1e-8)
            update_best(h_polished)
        except Exception:
            pass
    
    # Phase 8: Final high-precision polish with exact value
    if deadline_ok():
        try:
            # Use a finer coordinate search with exact value
            h_final = best_h.copy()
            for i in range(N):
                if not deadline_ok():
                    break
                best_local = h_final[i]
                best_local_val = _value(h_final)
                scale = max(0.005, abs(h_final[i]) * 0.05)
                for step in [-2*scale, -scale, -scale/3, scale/3, scale, 2*scale]:
                    h_test = h_final.copy()
                    h_test[i] = h_final[i] + step
                    v_test = _value(h_test)
                    if v_test < best_local_val - 1e-12:
                        best_local_val = v_test
                        best_local = h_test[i]
                h_final[i] = best_local
            update_best(h_final)
        except Exception:
            pass
    
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
