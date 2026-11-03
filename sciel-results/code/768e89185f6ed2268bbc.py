"""
Combined spectral-envelope + direct L-BFGS optimization for signed step functions
minimizing max|autoconv| / (sum x)^2.

Structural change: Instead of relying on a low-dimensional parametric spectral envelope
(which may miss optimal structures), we use a hybrid approach:
1. Generate diverse initializations (structured, random, banded, spectral-param)
2. Run L-BFGS-B with soft-max surrogate on each initialization
3. Refine with higher-beta L-BFGS
4. Peak-targeted coordinate descent polish
5. Final BFGS polish

The key insight from inspiration 1: direct optimization in x-space with diverse
initializations and proper soft-max gradients outperforms parametric spectral approaches
because the optimal structure is not well-captured by simple Gaussian/cosine envelopes.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 140
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

def _soft_max_abs_ac(ac, beta=100.0):
    """Compute soft-max of |ac| and weights."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(abs_ac)
    shifted = beta * (abs_ac - m)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max_val = m + (1.0 / beta) * np.log(sum_exp)
    return soft_max_val, weights

def _soft_score_grad(x, beta=100.0):
    """
    Compute soft-max surrogate score and its gradient.
    Returns (value, gradient).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    soft_max_val, weights = _soft_max_abs_ac(ac, beta)
    
    if soft_max_val < 1e-15:
        return 0.0, np.zeros_like(x)
    
    val = soft_max_val / (s * s)
    
    # Gradient computation using FFT
    L_fft = 1
    while L_fft < 2 * N - 1:
        L_fft *= 2
    
    # weights has length 2*N-1, pad to L_fft
    w_padded = np.zeros(L_fft)
    w_padded[:len(weights)] = weights
    
    W = np.fft.rfft(w_padded, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1: conv of weights with x (circular, but we take first N)
    term1 = np.fft.irfft(W * Xf, n=L_fft)[:N]
    
    # term2: conv of flipped weights with x
    w_flip = weights[::-1]
    w_flip_padded = np.zeros(L_fft)
    w_flip_padded[:len(w_flip)] = w_flip
    W_flip = np.fft.rfft(w_flip_padded, n=L_fft)
    term2 = np.fft.irfft(W_flip * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    # Full gradient: d(val)/dx = (1/s^2) * d(soft_max)/dx - 2*soft_max/s^3 * 1
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max_val / (s ** 3)) * np.ones(N)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(x)
    
    return val, grad

def _structured_init(freq, phase=0.0):
    """Create a structured initialization with a dominant frequency component."""
    t = np.arange(N)
    x = np.cos(2 * np.pi * freq * t / N + phase)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x = x / s
    return x

def _random_init(rng):
    """Create a random initialization."""
    x = rng.standard_normal(N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x = x / s
    return x

def _banded_init(bandwidth, rng):
    """Create an initialization with banded structure."""
    x = np.zeros(N)
    for i in range(N):
        for j in range(max(0, i - bandwidth), min(N, i + bandwidth + 1)):
            x[j] += rng.standard_normal()
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x = x / s
    return x

def _spectral_param_to_x(params, rng=None):
    """
    Convert spectral envelope parameters to a time-domain signal.
    Used for diverse initializations.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    
    a = np.clip(params[0], 0.1, 5.0)
    tilt = np.clip(params[1], -1.0, 1.0)
    
    k = np.arange(N)
    f = k / N
    
    mag_sq = np.exp(-a * f**2)
    mag_sq = mag_sq * (1.0 + tilt * f)
    mag_sq = np.maximum(mag_sq, 1e-10)
    
    idx = 2
    for _ in range(3):
        if idx + 2 > len(params):
            break
        phase_k = params[idx]
        amp_k = np.clip(params[idx + 1], -0.5, 0.5)
        freq_k = np.clip(params[idx + 2], 0.5, 5.0)
        idx += 3
        mag_sq = mag_sq * (1.0 + amp_k * np.cos(2 * np.pi * freq_k * f + phase_k))
        mag_sq = np.maximum(mag_sq, 1e-10)
    
    mag = np.sqrt(mag_sq)
    
    if idx < len(params):
        global_phase = params[idx]
    else:
        global_phase = 0.0
    
    X = mag * np.exp(1j * global_phase)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.01
        s = np.sum(x)
    x = x / s
    return x

def _lbfgs_optimize(x0, beta, maxiter, ftol, gtol, deadline_check=True):
    """Run L-BFGS-B optimization with soft-max surrogate."""
    def objective_and_grad(x_flat):
        val, g = _soft_score_grad(x_flat, beta)
        if not np.all(np.isfinite(g)):
            g = np.zeros_like(x_flat)
        if not np.isfinite(val):
            val = 1e10
        return val, g
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': ftol,
                'gtol': gtol,
                'maxfun': maxiter * 3
            }
        )
        x_opt = result.x
        sc = _objective(x_opt)
        return x_opt, sc
    except Exception:
        return x0, _objective(x0)

def _peak_targeted_polish(x, best_val, max_steps=2000):
    """
    Peak-targeted coordinate descent: identify dominant autoconvolution peaks
    and try to reduce them by adjusting contributing x values.
    """
    x = x.copy()
    current_val = best_val
    
    for step in range(max_steps):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        # Find top K peaks
        K = 10
        peak_indices = np.argsort(abs_ac)[-K:]
        
        improved = False
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
            
            # Try small adjustments
            for delta in [-0.05, -0.02, 0.02, 0.05]:
                trial = x.copy()
                trial[j_range] += delta
                val = _objective(trial)
                if val < current_val:
                    current_val = val
                    x = trial
                    improved = True
                    break
        
        if not improved and step > 100:
            break
    
    return x, current_val

def _random_perturbation_search(x, best_val, max_iters=5000, rng=None):
    """
    Random perturbation with smooth surrogate screening.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    x = x.copy()
    current_val = best_val
    step_size = 0.05
    stall_count = 0
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = x + perturbation
        
        val_smooth, _ = _soft_score_grad(x_trial, beta=50.0)
        
        if val_smooth < current_val * 0.999:
            val_exact = _objective(x_trial)
            if val_exact < current_val:
                current_val = val_exact
                x = x_trial
                stall_count = 0
                step_size = max(0.001, step_size * 0.9998)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 2000 == 0:
            step_size = min(0.5, step_size * 1.1)
    
    return x, current_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(123)
    
    # Phase 1: Generate diverse initializations
    inits = [np.ones(N)]
    
    # Structured initializations at different frequencies
    for freq in [1, 2, 3, 5, 7, 10, 13, 17, 21, 29]:
        inits.append(_structured_init(freq))
        inits.append(_structured_init(freq, phase=np.pi/2))
    
    # Random initializations
    for _ in range(6):
        inits.append(_random_init(rng))
    
    # Banded initializations
    for bw in [3, 5, 8, 12]:
        inits.append(_banded_init(bw, rng))
    
    # Spectral parameter initializations
    spectral_inits = [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0],
        [1.5, 0.5, 0.0, 0.2, 1.5, np.pi, -0.2, 2.5, np.pi/2, 0.1, 3.5, np.pi/4],
        [1.5, -0.5, np.pi, 0.2, 1.5, 0, -0.2, 2.5, np.pi, 0.1, 3.5, 0],
        [0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0],
    ]
    for p0 in spectral_inits:
        inits.append(_spectral_param_to_x(p0, rng))
    
    # More random spectral inits
    for _ in range(8):
        params = [
            rng.uniform(0.3, 4.0),
            rng.uniform(-0.8, 0.8),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi),
        ]
        inits.append(_spectral_param_to_x(params, rng))
    
    # Phase 2: L-BFGS optimization from each initialization
    for i, x0 in enumerate(inits):
        if _check_deadline():
            break
        
        # Use moderate beta for initial optimization
        x_opt, sc = _lbfgs_optimize(x0, beta=80.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        
        if sc < best_val:
            best_val = sc
            best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Refinement with higher beta on the best solution
    x_opt, sc = _lbfgs_optimize(best_x, beta=150.0, maxiter=500, ftol=1e-14, gtol=1e-12)
    if sc < best_val:
        best_val = sc
        best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Peak-targeted coordinate descent polish
    x_polished, val_polished = _peak_targeted_polish(best_x, best_val, max_steps=3000)
    if val_polished < best_val:
        best_val = val_polished
        best_x = x_polished.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Random perturbation search
    x_pert, val_pert = _random_perturbation_search(best_x, best_val, max_iters=8000, rng=rng)
    if val_pert < best_val:
        best_val = val_pert
        best_x = x_pert.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final high-beta L-BFGS polish
    x_final, sc_final = _lbfgs_optimize(best_x, beta=200.0, maxiter=400, ftol=1e-14, gtol=1e-12)
    if sc_final < best_val:
        best_val = sc_final
        best_x = x_final.copy()
    
    # Phase 7: One more peak-targeted polish
    if not _check_deadline():
        x_polished2, val_polished2 = _peak_targeted_polish(best_x, best_val, max_steps=1000)
        if val_polished2 < best_val:
            best_val = val_polished2
            best_x = x_polished2.copy()
    
    return best_x
