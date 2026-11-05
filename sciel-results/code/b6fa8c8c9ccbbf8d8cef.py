"""Combined program: Parameterized Global Search + Full-Space L-BFGS-B Refinement.

Strategy:
1. Global search using Differential Evolution on rich parameterized families:
   - Windowed Zadoff-Chu sequences.
   - Golay complementary-like pairs (sum of two chirps).
   - DCT spectral shaping.
2. Select the best candidate h.
3. Refine the candidate in the FULL 400-dimensional space using L-BFGS-B.
   - To handle scale invariance, we fix sum(h) = 1 by parameterizing h as 
     [x_1, ..., x_399, 1 - sum(x_1..x_399)].
   - Use a smooth log-sum-exp surrogate for the max of the convolution to 
     provide gradients.
4. Deadline guard ensures we return before 150s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # safety margin under 150s wall-clock

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
    """Smooth approximation of max of autoconvolution using log-sum-exp.
    Returns the value scaled by 2*N / (sum(h))^2.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    # log-sum-exp trick
    scaled = beta * (conv - m)
    # clip to avoid overflow
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

# --- Parameterized Generators from Inspiration 1 ---

def _windowed_zc_h(params):
    """Windowed Zadoff-Chu sequence."""
    phase, rate, amp, base, win_width, win_center = params
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    x = (rel_pos - win_center) / (0.5 * win_width)
    window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    return base + amp * window * np.cos(ang)

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

# DCT basis for spectral shaping
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

def _dct_obj(coeffs):
    h = _dct_h(coeffs)
    return _value(h)

def _dct_smooth_obj(coeffs, beta=50.0):
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

# --- Full-Space Refinement ---

def _reconstruct_h(x):
    """Reconstruct h from x (first N-1 components) with sum(h)=1."""
    h = np.zeros(N)
    h[:N-1] = x
    h[N-1] = 1.0 - np.sum(x)
    return h

def _full_space_obj(x, beta=50.0):
    """Objective for full-space optimization.
    x has length N-1. h is reconstructed such that sum(h)=1.
    """
    h = _reconstruct_h(x)
    # We can skip the division by (sum h)^2 since sum h = 1
    # But for safety, we compute the full value
    return _smooth_max_value(h, beta)

def _full_space_obj_exact(x):
    """Exact objective for full-space optimization."""
    h = _reconstruct_h(x)
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

    # Phase 1: Global Search on Parameterized Families
    
    # 1a: Windowed Zadoff-Chu
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),      # phase
                (-50.0, 50.0),        # rate
                (-3.0, 3.0),          # amp
                (-3.0, 3.0),          # base
                (0.3, 1.5),           # win_width
                (0.2, 0.8),           # win_center
            ]
            res = differential_evolution(
                lambda p: _value(_windowed_zc_h(p)),
                bounds_wzc,
                seed=13,
                maxiter=40,
                tol=1e-6,
                popsize=12,
                init='sobol',
                polish=False,
                update_interval=1
            )
            if res.x is not None:
                h = _windowed_zc_h(res.x)
                update_best(h)
        except Exception:
            pass

    # 1b: Golay Complementary Pairs
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),      # p1
                (-50.0, 50.0),        # r1
                (-3.0, 3.0),          # a1
                (-np.pi, np.pi),      # p2
                (-50.0, 50.0),        # r2
                (-3.0, 3.0),          # a2
                (-3.0, 3.0),          # base
            ]
            res = differential_evolution(
                lambda p: _value(_golay_h(p)),
                bounds_golay,
                seed=7,
                maxiter=40,
                tol=1e-6,
                popsize=12,
                init='sobol',
                polish=False,
                update_interval=1
            )
            if res.x is not None:
                h = _golay_h(res.x)
                update_best(h)
        except Exception:
            pass

    # 1c: DCT Spectral Shaping
    if deadline_ok():
        try:
            # Try a few random initializations for DCT
            rng = np.random.default_rng(42)
            for _ in range(3):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(12, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass

    # Phase 2: Full-Space Refinement
    # This is the structural change: optimizing all 400 variables directly.
    if deadline_ok():
        try:
            # Normalize best_h to sum=1
            s = np.sum(best_h)
            if s != 0:
                h_norm = best_h / s
            else:
                h_norm = np.ones(N) / N
            
            # Extract first N-1 components for optimization
            x0 = h_norm[:N-1].copy()
            
            # Use smooth max for gradients
            # Beta controls the sharpness of the max approximation.
            # Too small -> too smooth, misses the peak.
            # Too large -> gradients become noisy/stiff.
            # 50-100 is usually good.
            beta = 80.0
            
            res_full = minimize(
                lambda x: _full_space_obj(x, beta=beta),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res_full.x is not None:
                h_refined = _reconstruct_h(res_full.x)
                update_best(h_refined)
                
                # Second pass with sharper beta if time permits
                if deadline_ok() and (time.time() - _t0) < _DEADLINE - 20:
                    beta_sharp = 200.0
                    x0_sharp = h_refined[:N-1].copy()
                    res_full2 = minimize(
                        lambda x: _full_space_obj(x, beta=beta_sharp),
                        x0_sharp,
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if res_full2.x is not None:
                        h_refined2 = _reconstruct_h(res_full2.x)
                        update_best(h_refined2)
        except Exception:
            pass

    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    # Ensure sum is not zero for the evaluator
    if abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
        
    return best_h
