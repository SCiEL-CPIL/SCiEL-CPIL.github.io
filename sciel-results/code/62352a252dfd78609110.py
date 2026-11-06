"""Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu +
full-dimensional coordinate descent + smooth-max refinement.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional coordinate descent with smooth logsumexp surrogate to escape
  Fourier subspace restriction, allowing high-frequency corrections.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock
_K_DCT = 40
_K_DFT = 20

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

def _dft_smooth_obj(coeffs_base, beta=80.0):
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

def _full_dim_smooth_obj(h, beta=80.0):
    """Smooth objective for full-dimensional h."""
    return _smooth_max_value(h, beta)

def _full_dim_coord_descent(h0, max_iter=500, step_size=0.01, beta=80.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Escapes Fourier subspace restriction, allowing high-frequency corrections.
    Uses in-place updates to avoid repeated array copies.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        improved = False
        # Try random coordinates
        idx = np.random.choice(N, size=80, replace=False)
        for i in idx:
            old_val = h[i]
            # Try several step sizes
            for delta in [step_size, -step_size, 2*step_size, -2*step_size,
                         5*step_size, -5*step_size, 10*step_size, -10*step_size]:
                h[i] = old_val + delta
                v = _smooth_max_value(h, beta)
                if v < best_v:
                    best_v = v
                    improved = True
                    break
                h[i] = old_val
            if not improved:
                h[i] = old_val
        if not improved:
            # Try smaller steps
            for i in idx[:20]:
                old_val = h[i]
                for delta in [step_size*0.1, -step_size*0.1, step_size*0.5, -step_size*0.5]:
                    h[i] = old_val + delta
                    v = _smooth_max_value(h, beta)
                    if v < best_v:
                        best_v = v
                        improved = True
                        break
                    h[i] = old_val
                if not improved:
                    h[i] = old_val
            if not improved:
                break
    
    return h, best_v

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0
    
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
    
    rng = np.random.default_rng(42)
    
    # === Strategy 1: DCT spectral shaping ===
    for restart in range(10):
        if not deadline_ok():
            break
        c0 = rng.normal(0, 0.5, _K_DCT)
        res = minimize(_dct_smooth_obj, c0, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.x is not None:
            h = _dct_h(res.x)
            update_best(h)
    
    # === Strategy 2: Golay complementary pair ===
    for restart in range(8):
        if not deadline_ok():
            break
        p0 = [rng.uniform(-np.pi, np.pi), rng.uniform(20, 80), rng.uniform(0.3, 1.0),
              rng.uniform(-np.pi, np.pi), rng.uniform(-80, -20), rng.uniform(0.3, 1.0),
              rng.uniform(0.5, 2.0)]
        res = minimize(_golay_smooth_obj, p0, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.x is not None:
            h = _golay_complementary_h(res.x)
            update_best(h)
    
    # === Strategy 3: Windowed Zadoff-Chu ===
    for restart in range(5):
        if not deadline_ok():
            break
        p0 = [rng.uniform(-np.pi, np.pi), rng.uniform(30, 100), rng.uniform(0.5, 1.5),
              rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.0), rng.uniform(0.3, 0.7),
              rng.choice([0.0, 1.0])]
        res = minimize(_windowed_zc_smooth_obj, p0, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.x is not None:
            h = _windowed_zadoff_chu_h(res.x)
            update_best(h)
    
    # === Strategy 4: DFT refinement on best candidate ===
    coeffs, base_val = _project_dft(best_h)
    coeffs_base = np.concatenate([coeffs, [base_val]])
    res = minimize(_dft_smooth_obj, coeffs_base, method='L-BFGS-B',
                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
    if res.x is not None:
        h = _dft_h(res.x[:-1], res.x[-1])
        update_best(h)
    
    # === Strategy 5: Full-dimensional coordinate descent ===
    if deadline_ok():
        h_cd, _ = _full_dim_coord_descent(best_h, max_iter=500, step_size=0.01, beta=80.0)
        update_best(h_cd)
    
    # === Strategy 6: Final polish with Nelder-Mead on small perturbation space ===
    if deadline_ok():
        def perturb_obj(deltas):
            h_trial = best_h + deltas
            return _smooth_max_value(h_trial, beta=100.0)
        
        for restart in range(3):
            if not deadline_ok():
                break
            d0 = rng.normal(0, 0.05, N)
            res = minimize(perturb_obj, d0, method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 200})
            if res.x is not None:
                h = best_h + res.x
                update_best(h)
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
