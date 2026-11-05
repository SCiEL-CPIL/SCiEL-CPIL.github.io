"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B (enforces nonzero sum via base offset)
- Secondary: Dual Zadoff-Chu complementary pair (two orthogonal chirps with independent phase/amplitude)
- Tertiary: Windowed Zadoff-Chu with global search
- Final: High-dimensional L-BFGS-B polish with progressive smooth max
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock
_K_DCT = 30  # number of DCT basis vectors for spectral shaping
_K_DFT = 15  # number of DFT frequency components for refinement

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
    # Smooth max via log-sum-exp (scaled)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -80, 80)
    smooth_m = m + np.log(np.sum(np.exp(shifted))) / beta
    return abs(2.0 * N * smooth_m / (s * s))

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

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)  # (N, K_DCT)

def _dct_h(coeffs, base_val=1.0):
    """Generate h from DCT coefficients + base offset to ensure nonzero sum."""
    h = base_val + _DCT_BASIS @ coeffs
    return h

def _dct_smooth_obj(params, beta=50.0):
    """Smooth objective for DCT coefficients + base. params: [coeffs..., base]"""
    coeffs = params[:-1]
    base_val = params[-1]
    h = _dct_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _dual_zc_h(params):
    """Dual Zadoff-Chu complementary pair:
    h[i] = base + a1*cos(2*pi*r1*i*(i+1)/N + p1) + a2*cos(2*pi*r2*i*(i+1)/N + p2)
    
    params: [p1, r1, a1, p2, r2, a2, base]
    The two chirps with different rates can have complementary autocorrelation properties.
    """
    p1, r1, a1, p2, r2, a2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * r1 * i * (i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * i * (i + 1) / N + p2
    h = base + a1 * np.cos(ang1) + a2 * np.cos(ang2)
    return h

def _dual_zc_obj(params):
    """Objective for dual Zadoff-Chu parameters."""
    h = _dual_zc_h(params)
    return _value(h)

def _dual_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for dual Zadoff-Chu parameters."""
    h = _dual_zc_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu: h[i] = base + amp * window[i] * cos(2*pi*rate*i*(i+1)/N + phase)."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    x = (rel_pos - win_center) / (0.5 * win_width)
    window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    
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

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K_DFT)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_smooth_obj(params, beta=40.0):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = params[:-1]
    base_val = params[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _admm_flat_autocorr(h_init, max_iter=200, rho=1.0, sigma=1.0, tol=1e-6, deadline_check=None):
    """
    ADMM to minimize ||conv(h,h) - C*1||_1 subject to sum(h)=1.
    This drives the entire autocorrelation profile flat.
    """
    h = h_init.copy()
    s = np.sum(h)
    h = h / s  # normalize to sum=1
    
    # Initialize C as the mean of the autoconvolution
    conv = _fft_convolve(h)
    C = np.mean(conv)
    
    # Initialize auxiliary variables
    z = conv.copy()  # z = conv(h,h)
    u = np.zeros_like(conv)  # scaled dual variable
    
    for it in range(max_iter):
        if deadline_check and not deadline_check():
            break
            
        # h-update: minimize ||conv(h,h) - z + u||_1 + (rho/2)||h - (z - u)||^2
        # This is complex; instead use a simpler approach:
        # h-update: minimize (1/2)||conv(h,h) - (z - u)||^2 + (rho/2)||h - h_old||^2
        # which is a quadratic in h (since conv is bilinear)
        
        # For simplicity, use a gradient step on the h-update
        # grad_h = 2 * (conv(h,h) - (z - u)) * h  (for the quadratic)
        conv_current = _fft_convolve(h)
        target = z - u
        grad = 2.0 * (conv_current - target) * h
        h_new = h - (1.0 / rho) * grad
        
        # Project onto sum=1
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-12:
            h_new = h_new / s_new
        
        # z-update: prox of (1/rho)*||.||_1
        # z = soft_threshold(conv(h_new) + u, 1/rho)
        conv_new = _fft_convolve(h_new)
        z_new = np.sign(conv_new + u) * np.maximum(np.abs(conv_new + u) - 1.0/rho, 0.0)
        
        # C-update: C = mean(z_new) (to keep the target centered)
        C = np.mean(z_new)
        
        # u-update
        u = u + conv_new - z_new
        
        # Check convergence
        h_change = np.linalg.norm(h_new - h) / (np.linalg.norm(h) + 1e-12)
        z_change = np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-12)
        h = h_new
        z = z_new
        
        if h_change < tol and z_change < tol:
            break
    
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
    
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
        if v < best_v and np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-12:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (primary)
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(8):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(12, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 0.8, size=n_active)
                base_init = 1.0 + rng.normal(0, 0.1)
                
                params_init = np.concatenate([coeffs, [base_init]])
                
                res = minimize(
                    lambda p: _dct_smooth_obj(p, beta=50.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(-4.0, 4.0)] * _K_DCT + [(0.1, 10.0)],
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9, 'iprint': 0}
                )
                if res.fun < np.inf:
                    coeffs_opt = res.x[:-1]
                    base_opt = res.x[-1]
                    h_candidate = _dct_h(coeffs_opt, base_opt)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Dual Zadoff-Chu complementary pair optimization
    if deadline_ok():
        try:
            rng = np.random.default_rng(789)
            for restart in range(5):
                if not deadline_ok():
                    break
                # Initialize with different rate ratios for complementary behavior
                r1_init = rng.uniform(0.3, 3.0)
                r2_init = r1_init + rng.uniform(0.5, 2.0)  # distinct rate
                
                params_init = np.array([
                    rng.uniform(-np.pi, np.pi),
                    r1_init,
                    rng.uniform(-3.0, 3.0),
                    rng.uniform(-np.pi, np.pi),
                    r2_init,
                    rng.uniform(-3.0, 3.0),
                    1.0,
                ])
                
                res = minimize(
                    lambda p: _dual_zc_smooth_obj(p, beta=50.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.1, 10.0), (-5.0, 5.0),
                            (-np.pi, np.pi), (0.1, 10.0), (-5.0, 5.0),
                            (0.1, 10.0)],
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'iprint': 0}
                )
                if res.fun < np.inf:
                    h_candidate = _dual_zc_h(res.x)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu with differential evolution
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-5.0, 5.0),
                (-10.0, 10.0),
                (0.1, 10.0),
                (0.3, 1.5),
                (0.2, 0.8),
            ]
            
            res_de = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=42,
                maxiter=40,
                popsize=10,
                tol=1e-5,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < np.inf:
                h_candidate = _windowed_zadoff_chu_h(res_de.x)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 4: DFT refinement from best_h
    if deadline_ok():
        try:
            base_val = np.mean(best_h)
            h_centered = best_h - base_val
            coeffs_init, _, _, _ = np.linalg.lstsq(_BASIS_T, h_centered, rcond=None)
            
            params_init = np.concatenate([coeffs_init, [base_val]])
            bounds_dft = [(-10.0, 10.0)] * (2 * _K_DFT) + [(0.1, 10.0)]
            
            res_dft = minimize(
                lambda p: _dft_smooth_obj(p, beta=40.0),
                params_init,
                method='L-BFGS-B',
                bounds=bounds_dft,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            if res_dft.fun < np.inf:
                coeffs_opt = res_dft.x[:-1]
                base_opt = res_dft.x[-1]
                h_candidate = _dft_h(coeffs_opt, base_opt)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 5: ADMM to flatten autocorrelation profile
    if deadline_ok():
        try:
            h_admm = _admm_flat_autocorr(best_h, max_iter=100, rho=0.5, tol=1e-5, deadline_check=deadline_ok)
            update_best(h_admm)
        except Exception:
            pass
    
    # Phase 6: Final high-dimensional polish with progressive smooth max
    if deadline_ok():
        try:
            # Start with moderate smoothness, tighten progressively
            beta_schedule = [20.0, 10.0, 5.0]
            current_h = best_h.copy()
            for beta in beta_schedule:
                if not deadline_ok():
                    break
                res_final = minimize(
                    lambda hv: _smooth_max_value(hv, beta=beta),
                    current_h,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0}
                )
                if res_final.fun < np.inf:
                    h_final = res_final.x
                    v_final = _value(h_final)
                    if v_final < best_v and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                        best_h = h_final
                        best_v = v_final
                        current_h = h_final.copy()
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
