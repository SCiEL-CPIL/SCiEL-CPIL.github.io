"""
Spectral-envelope optimization for signed step functions minimizing max|autoconv|/(sum x)^2.

Structural change: Instead of optimizing x directly with soft-max surrogates, we work in the
spectral domain. The autoconvolution in frequency is |X(f)|^2, and the peak constraint becomes
a constraint on the time-domain signal whose spectrum is |X(f)|^2. We parameterize X(f) as a
smooth spectral envelope and optimize its parameters to minimize the H-infinity-like norm
of the autoconvolution relative to the squared integral.

Key insight: If we write x as the IFFT of X, then autoconv = IFFT(|X|^2). The max|autoconv|
is the peak of a signal whose spectrum is |X|^2. By the uncertainty principle, spreading
energy across frequencies reduces peak. We optimize a parametric family of |X|^2 that
minimizes the peak of IFFT(|X|^2) / (sum x)^2.

We also add a direct SDP-like approach: iterate on the constraint |ac[n]| <= t * s^2 for all n,
using projected gradient on x with the constraint set {x : max|ac(x)| <= t*s^2}.
"""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 400
START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_param_to_x(params):
    """
    Convert spectral envelope parameters to a time-domain signal.
    params = [a, tilt, gamma, phase1, amp1, phase2, amp2, ...]
    
    We model |X(f)|^2 as a product of smooth factors:
      |X(f)|^2 = exp(-a * f^2) * (1 + tilt * f) * prod_k (1 + amp_k * cos(2*pi*f*k + phase_k))
    
    Then x = IFFT(sqrt(|X|^2) * exp(i*phase_total))
    """
    a = np.clip(params[0], 0.1, 5.0)
    tilt = np.clip(params[1], -1.0, 1.0)
    
    # Frequency grid for the spectral envelope
    k = np.arange(N)
    f = k / N  # normalized frequency [0, 1)
    
    # Base Gaussian envelope
    mag_sq = np.exp(-a * f**2)
    mag_sq = mag_sq * (1.0 + tilt * f)
    mag_sq = np.maximum(mag_sq, 1e-10)
    
    # Additional spectral shaping factors
    idx = 2
    for _ in range(3):  # up to 3 additional factors
        if idx + 2 > len(params):
            break
        phase_k = params[idx]
        amp_k = np.clip(params[idx + 1], -0.5, 0.5)
        freq_k = np.clip(params[idx + 2], 0.5, 5.0)
        idx += 3
        mag_sq = mag_sq * (1.0 + amp_k * np.cos(2 * np.pi * freq_k * f + phase_k))
        mag_sq = np.maximum(mag_sq, 1e-10)
    
    # Take square root to get |X|
    mag = np.sqrt(mag_sq)
    
    # Add a global phase for the signal
    if idx < len(params):
        global_phase = params[idx]
    else:
        global_phase = 0.0
    
    X = mag * np.exp(1j * global_phase)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    return x

def _spectral_objective(params):
    """Objective for spectral parameter optimization."""
    x = _spectral_param_to_x(params)
    return _objective(x)

def _projected_gradient_step(x, t, s2, lr=0.01, max_proj_iters=10):
    """
    Projected gradient step for the constraint max|ac| <= t * s^2.
    Uses subgradient of max|ac| for the projection.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Find which constraints are violated
    violations = abs_ac - t * s2
    active = violations > 0
    
    if not np.any(active):
        return x
    
    # Subgradient of max|ac| w.r.t. x
    # d(ac[n])/dx[j] = x[n-j] + x[j-n] (with boundary handling)
    # For simplicity, use the top violator
    n_top = np.argmax(abs_ac)
    sign_ac = np.sign(ac[n_top])
    
    # Gradient of |ac[n_top]| w.r.t. x[j]
    # ac[n_top] = sum_j x[j] * x[n_top - j]
    # d|ac[n_top]|/dx[j] = sign_ac * (x[n_top-j] + x[j-n_top])
    j_range = np.arange(max(0, n_top - N + 1), min(N, n_top + 1))
    partner = n_top - j_range
    valid = (partner >= 0) & (partner < N)
    j_range = j_range[valid]
    partner = partner[valid]
    
    grad = np.zeros(N)
    grad[j_range] += sign_ac * x[partner]
    grad[partner] += sign_ac * x[j_range]
    
    # Gradient descent step
    x_new = x - lr * grad / (s2 + 1e-10)
    
    return x_new

def _sdp_relaxation(x0, max_iters=200, lr_init=0.01):
    """
    Iterative SDP-like relaxation: alternate between
    1. Fixing t and optimizing x to minimize max|ac|/s^2
    2. Updating t to the current max|ac|/s^2
    """
    x = x0.copy()
    best_val = _objective(x)
    best_x = x.copy()
    
    s = np.sum(x)
    t = np.max(np.abs(_autoconv_fft(x))) / (s * s)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Update t
        ac = _autoconv_fft(x)
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        t = np.max(np.abs(ac)) / (s * s)
        
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Gradient step to reduce max|ac|
        abs_ac = np.abs(ac)
        n_top = np.argmax(abs_ac)
        sign_ac = np.sign(ac[n_top])
        
        j_range = np.arange(max(0, n_top - N + 1), min(N, n_top + 1))
        partner = n_top - j_range
        valid = (partner >= 0) & (partner < N)
        j_range = j_range[valid]
        partner = partner[valid]
        
        grad = np.zeros(N)
        grad[j_range] += sign_ac * x[partner]
        grad[partner] += sign_ac * x[j_range]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        # Adaptive learning rate
        lr = lr_init / (1.0 + 0.01 * it)
        x_new = x - lr * grad
        
        # Check if improvement
        val_new = _objective(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
            x = x_new
        else:
            # Try smaller step
            x_new = x - 0.5 * lr * grad
            val_new = _objective(x_new)
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
                x = x_new
            else:
                break
    
    return best_x, best_val

def _multi_scale_search(x0):
    """
    Multi-scale optimization: start with a coarse version (average over blocks),
    optimize, then refine at full resolution.
    """
    # Coarse: average over blocks of size B
    B = 8
    n_coarse = N // B
    x_coarse = np.zeros(n_coarse)
    for i in range(n_coarse):
        x_coarse[i] = np.mean(x0[i*B:(i+1)*B])
    
    # Optimize coarse
    def _coarse_obj(params):
        x_c = _spectral_param_to_x(params)
        # Upsample to full resolution
        x_full = np.zeros(N)
        for i in range(n_coarse):
            x_full[i*B:(i+1)*B] = x_c[i]
        return _objective(x_full)
    
    # Try a few parameter initializations for coarse
    best_coarse_val = _objective(x0)
    best_coarse_x = x0.copy()
    
    rng = np.random.default_rng(7)
    for trial in range(10):
        if _check_deadline():
            break
        params = np.array([
            rng.uniform(0.5, 3.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.3, 0.3), rng.uniform(1, 4),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.3, 0.3), rng.uniform(1, 4),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.3, 0.3), rng.uniform(1, 4),
            rng.uniform(0, 2*np.pi),
        ])
        
        try:
            result = minimize(_coarse_obj, params, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_coarse_val:
                best_coarse_val = result.fun
                x_c = _spectral_param_to_x(result.x)
                x_full = np.zeros(N)
                for i in range(n_coarse):
                    x_full[i*B:(i+1)*B] = x_c[i]
                best_coarse_x = x_full
        except Exception:
            pass
    
    # Refine at full resolution
    x_refined, val_refined = _sdp_relaxation(best_coarse_x, max_iters=100)
    return x_refined, val_refined

def _soft_max_objective_grad(x, beta=50.0):
    """Soft-max surrogate with gradient, optimized for speed."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    shifted = beta * (abs_ac - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient
    g = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Spectral parameter optimization (main structural change)
    # Optimize the spectral envelope parameters
    rng = np.random.default_rng(123)
    
    spectral_inits = []
    # Hand-crafted initializations
    spectral_inits.append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    spectral_inits.append([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0])
    spectral_inits.append([1.5, 0.5, 0.0, 0.2, 1.5, np.pi, -0.2, 2.5, np.pi/2, 0.1, 3.5, np.pi/4])
    spectral_inits.append([1.5, -0.5, np.pi, 0.2, 1.5, 0, -0.2, 2.5, np.pi, 0.1, 3.5, 0])
    spectral_inits.append([0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    spectral_inits.append([3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0])
    
    # Random initializations
    for _ in range(15):
        params = [
            rng.uniform(0.3, 4.0),
            rng.uniform(-0.8, 0.8),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi),
        ]
        spectral_inits.append(params)
    
    for p0 in spectral_inits:
        if _check_deadline():
            break
        try:
            result = minimize(_spectral_objective, p0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_param_to_x(result.x)
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: SDP-like relaxation from best spectral solution
    x_sdp, val_sdp = _sdp_relaxation(best_x, max_iters=300)
    if val_sdp < best_val:
        best_val = val_sdp
        best_x = x_sdp
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Soft-max L-BFGS refinement
    def _smooth_obj(x):
        val, grad = _soft_max_objective_grad(x, beta=50.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Multi-scale refinement
    x_ms, val_ms = _multi_scale_search(best_x)
    if val_ms < best_val:
        best_val = val_ms
        best_x = x_ms
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-beta L-BFGS polish
    def _high_beta_obj(x):
        val, grad = _soft_max_objective_grad(x, beta=100.0)
        return val, grad
    
    try:
        result = minimize(_high_beta_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Vectorized peak-targeted polish
    ac_current = _autoconv_fft(best_x)
    s_current = np.sum(best_x)
    abs_ac_current = np.abs(ac_current)
    
    # Find top peaks
    K = 30
    peak_indices = np.argsort(abs_ac_current)[-K:]
    
    for peak_idx in peak_indices:
        if _check_deadline():
            break
        
        j_range = np.arange(max(0, peak_idx - N + 1), min(N, peak_idx + 1))
        if len(j_range) == 0:
            continue
        
        partner_indices = peak_idx - j_range
        valid = (partner_indices >= 0) & (partner_indices < N)
        j_range = j_range[valid]
        partner_indices = partner_indices[valid]
        
        if len(j_range) == 0:
            continue
        
        # Vectorized trial: try adjusting all contributing x[j] simultaneously
        for delta in [-0.1, -0.05, 0.05, 0.1]:
            trial = best_x.copy()
            trial[j_range] += delta
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Random perturbation with smooth surrogate screening
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    
    for iteration in range(30000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val_smooth, _ = _soft_max_objective_grad(x_trial, beta=50.0)
        
        if val_smooth < best_val * 0.999:
            val_exact = _objective(x_trial)
            if val_exact < best_val:
                best_val = val_exact
                best_x = x_trial.copy()
                stall_count = 0
                step_size = max(0.001, step_size * 0.9998)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 5000 == 0:
            step_size = min(0.5, step_size * 1.05)
    
    # Phase 8: Final BFGS polish
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS',
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
