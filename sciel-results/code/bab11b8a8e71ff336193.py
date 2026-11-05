import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock

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

# --- Square-wave + polynomial window representation ---
# h[i] = A * sign(sin(2*pi*(i+0.5)/N)) * P(i/N)
# where P(x) is a low-order polynomial window.
# We parameterize P(x) = sum_{k=0}^{K} c_k * x^k
# and optimize A and c_k.

_K_POLY = 6  # polynomial order (up to degree 6)

def _sqwave_poly_h(params):
    """
    h[i] = A * sq[i] * P(x_i)
    where sq[i] = sign(sin(2*pi*(i+0.5)/N)) (square wave, +1 first half, -1 second half)
    P(x) = sum_{k=0}^{K} c_k * x^k, x = i/(N-1)
    params = [A, c_0, c_1, ..., c_K]
    """
    A = params[0]
    coeffs = params[1:]
    i = np.arange(N, dtype=float)
    x = i / (N - 1)
    # Square wave: +1 for first half, -1 for second half
    sq = np.where(i < N / 2, 1.0, -1.0)
    # Evaluate polynomial
    P = np.zeros(N)
    xk = np.ones(N)
    for k, c in enumerate(coeffs):
        P += c * xk
        xk *= x
    return A * sq * P

def _sqwave_poly_obj(params):
    h = _sqwave_poly_h(params)
    return _value(h)

def _sqwave_poly_smooth_obj(params, beta=80.0):
    h = _sqwave_poly_h(params)
    return _smooth_max_value(h, beta)

# --- DCT basis for spectral shaping ---
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

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# --- DFT basis for refinement ---
_K_DFT = 20
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    return base_val + _basis.T @ coeffs

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

# --- Windowed Zadoff-Chu ---
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

def _windowed_zc_smooth_obj(params, beta=80.0):
    h = _windowed_zc_h(params)
    return _smooth_max_value(h, beta)

# --- Golay complementary pair ---
def _golay_h(params):
    """
    Golay complementary pair: h = base + a1*cos(q1) + a2*cos(q2)
    where q1, q2 are quadratic phases (chirps).
    """
    p1, r1, a1, p2, r2, a2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * r1 * i * (i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * i * (i + 1) / N + p2
    return base + a1 * np.cos(ang1) + a2 * np.cos(ang2)

def _golay_obj(params):
    h = _golay_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=80.0):
    h = _golay_h(params)
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
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: Square-wave + polynomial window global search (primary structural idea)
    if deadline_ok():
        try:
            # params: [A, c0, c1, ..., c6]
            bounds_sp = [(0.5, 5.0)]  # A
            for k in range(_K_POLY + 1):
                bounds_sp.append((-3.0, 3.0))
            
            res = differential_evolution(
                _sqwave_poly_obj, bounds_sp, seed=11, maxiter=50, tol=1e-8,
                popsize=15, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _sqwave_poly_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: Local refinement of square-wave + polynomial window with smooth max
    if deadline_ok():
        try:
            # Start from the best found in Phase 1, or a reasonable default
            x0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # Try multiple restarts with different initial conditions
            rng = np.random.default_rng(42)
            for restart in range(5):
                if not deadline_ok():
                    break
                x0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for k in range(1, _K_POLY + 1):
                    x0[1 + k] = rng.normal(0, 0.5)
                res = minimize(_sqwave_poly_smooth_obj, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _sqwave_poly_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.3, 1.5),
                (0.2, 0.8),
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
            res = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=40, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: DCT spectral shaping with multiple restarts
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(6):
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
    
    # Phase 6: DFT refinement with smooth max
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K_DFT)
            for k in range(1, _K_DFT + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                A_k = np.column_stack([np.cos(angle), np.sin(angle)])
                target = h_best - mean_h
                sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                coeffs_init[2 * (k - 1)] = sol[0]
                coeffs_init[2 * (k - 1) + 1] = sol[1]
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(_dft_smooth_obj, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Full-dimensional coordinate descent with smooth log-sum-exp surrogate.
    # This escapes the band-limited Fourier/DCT subspace and allows high-frequency
    # corrections that flatten the autocorrelation peak more effectively.
    if deadline_ok():
        try:
            h_cd = best_h.copy()
            v_cd = _value(h_cd)
            # Adaptive step size starting from the signal scale
            step = 0.05 * max(1.0, float(np.max(np.abs(h_cd))))
            beta = 80.0
            n_iter_cd = 0
            max_cd_iters = 6000
            while deadline_ok() and n_iter_cd < max_cd_iters:
                # Random subset of indices to update per sweep (full-dim but budgeted)
                idx = np.random.default_rng(n_iter_cd).permutation(N)
                improved = False
                for i in idx:
                    if not deadline_ok():
                        break
                    h_try = h_cd.copy()
                    h_try[i] += step
                    v_try = _smooth_max_value(h_try, beta)
                    if v_try < v_cd:
                        h_cd = h_try
                        v_cd = v_try
                        improved = True
                    else:
                        h_try = h_cd.copy()
                        h_try[i] -= step
                        v_try = _smooth_max_value(h_try, beta)
                        if v_try < v_cd:
                            h_cd = h_try
                            v_cd = v_try
                            improved = True
                n_iter_cd += 1
                if not improved:
                    # Shrink step and retry a few times before giving up
                    step *= 0.5
                    if step < 1e-6:
                        break
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 8: Final DCT polish with tighter tolerance
    if deadline_ok():
        try:
            h_best = best_h
            coeffs_init = _DCT_BASIS.T @ h_best
            res = minimize(_dct_obj, coeffs_init, method='L-BFGS-B',
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 9: High-precision full-dimensional polish on the exact objective.
    # Uses a small-step coordinate ascent on the exact (nonsmooth) value to
    # squeeze out the last bit after the smooth surrogate converges.
    if deadline_ok():
        try:
            h_pol = best_h.copy()
            v_pol = _value(h_pol)
            step = 0.01 * max(1.0, float(np.max(np.abs(h_pol))))
            for _ in range(3000):
                if not deadline_ok():
                    break
                idx = np.random.default_rng(_).permutation(N)
                improved = False
                for i in idx:
                    if not deadline_ok():
                        break
                    h_try = h_pol.copy()
                    h_try[i] += step
                    v_try = _value(h_try)
                    if v_try < v_pol:
                        h_pol = h_try
                        v_pol = v_try
                        improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
            update_best(h_pol)
        except Exception:
            pass

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
