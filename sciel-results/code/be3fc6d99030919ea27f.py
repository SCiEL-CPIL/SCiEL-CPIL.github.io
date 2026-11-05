import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # safety margin under 150s wall-clock

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

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

def _surrogate_max(conv, tau=0.5):
    """Smooth approximation of max(conv) using logsumexp."""
    m = np.max(conv)
    if m > 1e10:
        return m
    scaled = (conv - m) / tau
    scaled = np.clip(scaled, -50, 50)
    return m + tau * np.log(np.mean(np.exp(scaled)))

def _surrogate_objective(h, tau=0.5):
    """Smooth surrogate: normalize sum to N, minimize surrogate_max(conv)/N."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    h_norm = h * N / s
    conv = _fft_convolve(h_norm)
    return _surrogate_max(conv, tau) / N

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

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# DFT basis for refinement
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

def _golay_h(params):
    """Golay complementary pair: h = base + a1*cos(q1) + a2*cos(q2)."""
    p1, r1, a1, p2, r2, a2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * r1 * i * (i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * i * (i + 1) / N + p2
    return base + a1 * np.cos(ang1) + a2 * np.cos(ang2)

def _golay_obj(params):
    h = _golay_h(params)
    return _value(h)

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

def _chirp_h(params):
    """Single chirp: h = base + amp * cos(phase + rate * (i/N)^2)."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * (i / N) ** 2
    return base + amp * np.cos(ang)

def _chirp_obj(params):
    h = _chirp_h(params)
    return _value(h)

def _barker_like_h(params):
    """Barker-like multi-harmonic sequence."""
    base = params[0]
    n_harmonics = 8
    amplitudes = params[1:1+n_harmonics]
    phases = params[1+n_harmonics:1+2*n_harmonics]
    i = np.arange(N, dtype=float)
    h = np.full(N, base)
    for k in range(1, n_harmonics + 1):
        h += amplitudes[k-1] * np.cos(2.0 * np.pi * k * i / N + phases[k-1])
    return h

def _barker_like_obj(params):
    h = _barker_like_h(params)
    return _value(h)

def _barker_like_smooth_obj(params, beta=80.0):
    h = _barker_like_h(params)
    return _smooth_max_value(h, beta)

def _admm_flat_profile(h, rho=1.0, max_iter=200, tol=1e-8):
    """
    ADMM to minimize ||conv(h,h) - C*1||_1 subject to sum(h) = N.
    This drives the entire autocorrelation profile toward a constant.
    
    We use the epigraph formulation:
    min  sum_j |conv_j - C|
    s.t. conv = conv(h,h), sum(h) = N
    
    ADMM splits: 
    - z = conv(h,h) (convolution constraint)
    - u = C (constant target)
    - sum(h) = N (linear constraint via projection)
    
    Simplified: alternate between:
    1. h-update: minimize rho/2 * ||conv(h,h) - z + u||^2 + lambda/2 * (sum(h)-N)^2
       (solved via gradient descent on h)
    2. z-update: z = soft_threshold(conv(h,h) + u, 1/rho)  (prox of L1)
    3. u-update: u = u + conv(h,h) - z
    4. C-update: C = mean(z)  (target constant)
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h
    h = h * N / s
    
    z = _fft_convolve(h)
    C = np.mean(z)
    u = np.zeros_like(z)
    
    for it in range(max_iter):
        # h-update: gradient descent on rho/2 * ||conv(h,h) - z + u||^2 + (lambda/2)*(sum(h)-N)^2
        # Gradient of conv(h,h) w.r.t. h[i] involves correlation
        # Use a few gradient steps
        for _ in range(3):
            conv_h = _fft_convolve(h)
            diff = conv_h - z + u
            # Gradient of 0.5*||conv(h,h)||^2 w.r.t. h = convolve(h, conv(h,h))
            # Actually, d/dh_i [0.5 * sum_j conv_j^2] where conv = h*h
            # = sum_j conv_j * d(conv_j)/d(h_i)
            # conv_j = sum_k h_k h_{j-k}
            # d(conv_j)/d(h_i) = h_{j-i} + h_{j-N+1+i} (for full conv)
            # This is expensive; use FFT-based approach
            # Alternatively, just use the fact that for the h-update we can use:
            # grad_i = sum_j diff_j * (h_{j-i} + h_{j-N+1+i})
            # Which is essentially convolve(diff, h)
            grad = np.zeros(N)
            # Use FFT for the gradient computation
            # grad[i] = sum_j diff[j] * h[j-i] (circular-ish)
            # Actually for full convolution:
            # Let's just do a simple approach: perturb and compute
            # For speed, use the fact that grad = convolve(diff, h) appropriately
            # diff has length 2N-1, h has length N
            # grad[i] = sum_j diff[j] * (h[j-i] if 0<=j-i<N else 0)
            # This is cross-correlation
            # Use FFT
            n_fft = 2 * N
            D = np.fft.rfft(diff, n=n_fft)
            H = np.fft.rfft(h, n=n_fft)
            grad_full = np.fft.irfft(D * H, n=n_fft)
            # The gradient for full convolution conv(h,h)[j] = sum_k h[k]*h[j-k]
            # d/dh[i] = sum_j diff[j] * (h[j-i] + h[j-(N-1)-i + (N-1)])
            # For simplicity, use the first N elements
            grad = grad_full[:N]
            
            # Add constraint gradient: lambda * (sum(h) - N) / N for each component
            lam = 1.0
            grad += lam * (np.sum(h) - N) / N
            
            # Gradient step
            lr = 0.01
            h = h - lr * grad
        
        # Ensure sum(h) = N
        s = np.sum(h)
        if abs(s) > 1e-6:
            h = h * N / s
        
        # z-update: prox of (1/rho)*||.||_1
        conv_h = _fft_convolve(h)
        z_old = z.copy()
        z = np.sign(conv_h + u) * np.maximum(np.abs(conv_h + u) - 1.0/rho, 0.0)
        
        # u-update
        u = u + conv_h - z
        
        # C-update
        C = np.mean(z)
        
        # Check convergence
        if np.max(np.abs(z - z_old)) < tol:
            break
    
    return h

def _coordinate_update_fast(h, i, tau=0.5, max_iter=30):
    """Fast coordinate update using precomputed structure."""
    h_old = h[i]
    h_fixed = h.copy()
    
    def obj(delta):
        h_new = h_fixed.copy()
        h_new[i] = h_old + delta
        return _surrogate_objective(h_new, tau)
    
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
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with multiple restarts
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
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair global search
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
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_h(res.x)
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
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=35, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _windowed_zc_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Chirp global search
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _chirp_obj, bounds_chirp, seed=31, maxiter=35, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: ADMM flat-profile optimization on best candidate
    if deadline_ok():
        try:
            h_admm = _admm_flat_profile(best_h, rho=1.0, max_iter=150, tol=1e-7)
            update_best(h_admm)
            
            # Try with different rho
            h_admm2 = _admm_flat_profile(best_h, rho=0.5, max_iter=100, tol=1e-7)
            update_best(h_admm2)
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
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Direct coordinate descent with smooth surrogate (from inspiration 1)
    if deadline_ok():
        try:
            h_curr = best_h.copy()
            s = np.sum(h_curr)
            if abs(s) > 1e-6:
                h_curr = h_curr * N / s
            
            rng = np.random.default_rng(99)
            taus = [0.3, 0.15, 0.08]
            
            for tau in taus:
                if not deadline_ok():
                    break
                
                n_passes = 2
                for p in range(n_passes):
                    if not deadline_ok():
                        break
                    
                    order = rng.permutation(N)
                    # Only update a subset per pass for speed
                    n_updates = min(100, N)
                    for idx in order[:n_updates]:
                        if not deadline_ok():
                            break
                        new_val = _coordinate_update_fast(h_curr, idx, tau=tau, max_iter=20)
                        h_curr[idx] = new_val
                    
                    # Check improvement
                    v_curr = _value(h_curr)
                    if v_curr < best_v:
                        update_best(h_curr)
                        h_curr = best_h.copy()
                        s = np.sum(h_curr)
                        if abs(s) > 1e-6:
                            h_curr = h_curr * N / s
        except Exception:
            pass
    
    # Phase 8: Final DCT polish with tighter tolerance
    if deadline_ok():
        try:
            h_best = best_h
            coeffs_init = _DCT_BASIS.T @ h_best
            res = minimize(_dct_obj, coeffs_init, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 9: Final coordinate descent polish
    if deadline_ok():
        try:
            h_curr = best_h.copy()
            s = np.sum(h_curr)
            if abs(s) > 1e-6:
                h_curr = h_curr * N / s
            
            rng = np.random.default_rng(77)
            tau_final = 0.05
            
            for _ in range(100):
                if not deadline_ok():
                    break
                
                idx = rng.integers(0, N)
                new_val = _coordinate_update_fast(h_curr, idx, tau=tau_final, max_iter=20)
                h_curr[idx] = new_val
                
                v_new = _value(h_curr)
                if v_new < best_v:
                    update_best(h_curr)
                    h_curr = best_h.copy()
                    s = np.sum(h_curr)
                    if abs(s) > 1e-6:
                        h_curr = h_curr * N / s
        except Exception:
            pass
    
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
