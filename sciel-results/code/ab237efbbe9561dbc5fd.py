"""Model-and-optimize: spectral phase retrieval + L-BFGS + peak-targeted polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient w.r.t. x.
    ac[k] = sum_i x[i] * x[i+k] for k=0..2N-2
    d(ac[k])/dx[i] = x[i+k] + x[i-k] (with x[j]=0 for j<0 or j>=N)
    So grad_x = ifft(fft(x) * fft(grad_ac)) where grad_ac is zero-padded
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # grad_x[i] = sum_k grad_ac[k] * (x[i+k] + x[i-k])
    # = ifft(fft(x) * fft(grad_ac_zero_padded))
    L_corr = _fft_len(3 * N)
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(grad_ac, n=L_corr)
    grad_x = np.fft.irfft(X_f * G_f, n=L_corr)[:N]
    
    return soft_val, grad_x

def _objective_only(x, beta):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
    else:
        soft_val = m
    return soft_val / (s * s)

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _peak_targeted_polish(x, max_iters=80, beta_start=100.0, beta_end=500.0):
    """
    Targeted polish: identify the largest autoconvolution peaks and
    apply focused gradient steps to reduce them with adaptive beta.
    """
    best_x = x.copy()
    best_val = _objective_only(best_x, beta_start)
    s_best = np.sum(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Adaptive beta: increase over iterations
        beta = beta_start * (beta_end / beta_start) ** (it / max(1, max_iters - 1))
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        
        # Find top K peaks and boost their weights
        K = min(10, len(abs_ac))
        top_idx = np.argsort(abs_ac)[-K:]
        peak_mask = np.zeros(len(abs_ac))
        peak_mask[top_idx] = 1.0
        
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
        
        weights = exps / np.sum(exps)
        # Boost weights at peak locations
        boost_factor = 3.0
        weights_boosted = weights * (1.0 + boost_factor * peak_mask)
        weights_boosted /= np.sum(weights_boosted)
        
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        grad_ac = weights_boosted * signs
        
        # Gradient w.r.t. x via correct FFT method
        L_corr = _fft_len(3 * N)
        X_f = np.fft.rfft(best_x, n=L_corr)
        G_f = np.fft.rfft(grad_ac, n=L_corr)
        grad_x = np.fft.irfft(X_f * G_f, n=L_corr)[:N]
        
        # Normalize gradient and take step
        g_norm = np.linalg.norm(grad_x)
        if g_norm < 1e-15:
            break
        grad_x_norm = grad_x / g_norm
        
        # Adaptive step size with backtracking
        step = 0.5 / (beta * m + 1.0)
        improved = False
        for _ in range(5):
            x_trial = best_x - step * grad_x_norm
            s_trial = np.sum(x_trial)
            if abs(s_trial) < 1e-6:
                step /= 2
                continue
            x_trial = x_trial / s_trial
            val_trial = _objective_only(x_trial, beta)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial.copy()
                improved = True
                break
            step /= 2
        
        if not improved:
            # Try a smaller random perturbation
            perturbation = np.random.default_rng(it).normal(0, 0.01, size=N)
            x_trial = best_x + perturbation
            s_trial = np.sum(x_trial)
            if abs(s_trial) > 1e-6:
                x_trial = x_trial / s_trial
                val_trial = _objective_only(x_trial, beta)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial.copy()
    
    return best_x, best_val

def _spectral_init_phase_optimized(rng, mode=0, iters=8):
    """
    Generate initial x via spectral phase retrieval, then refine phase 
    to minimize max|ac| (PAPR-like metric).
    """
    L = 2 * N - 1
    L_fft = _fft_len(L)
    
    # 1. Generate initial magnitude spectrum
    if mode == 0:
        # Flat-top with taper
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        # Raised cosine
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        # Banded: low frequency emphasis
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-2.0 * np.abs(freqs) * L_fft)
    else:
        # Random structure
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    
    # 2. Random phase
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0  # DC real
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x = np.fft.irfft(spectrum, n=L_fft)[:N]
    
    # Normalize sum to 1 for phase optimization stability
    s = np.sum(x)
    if abs(s) < 1e-10:
        return x
    x_norm = x / s
    
    # 3. Phase optimization: minimize max|ac|
    for _ in range(iters):
        if _check_deadline():
            break
        # Compute gradient of soft-max ac
        beta = 50.0
        val, grad = _soft_max_and_grad(x_norm, beta)
        
        # Update x_norm
        step = 1.0 / (beta * np.max(np.abs(x_norm)) + 1e-6)
        x_new = x_norm - step * grad
        
        # Re-project to fixed magnitude spectrum
        X_new = np.fft.rfft(x_new, n=L_fft)
        X_new = mag * (X_new / np.maximum(np.abs(X_new), 1e-10))
        X_new[0] = mag[0]  # Keep DC real
        
        x_norm = np.fft.irfft(X_new, n=L_fft)[:N]
        
        # Re-normalize sum to 1
        s_new = np.sum(x_norm)
        if abs(s_new) < 1e-10:
            break
        x_norm = x_norm / s_new

    return x_norm

def _amplitude_optimize(x_init, iters=15):
    """
    Optimize the amplitude spectrum (keeping phase fixed) to reduce max|ac|.
    This is a gradient-based approach in the frequency domain.
    """
    L = 2 * N - 1
    L_fft = _fft_len(L)
    
    X = np.fft.rfft(x_init, n=L_fft)
    mag = np.abs(X)
    phase = np.angle(X)
    mag[0] = np.abs(X[0])
    
    for i in range(iters):
        if _check_deadline():
            break
        
        # Reconstruct x from current mag and phase
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]  # Keep DC real
        x = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        x_norm = x / s
        
        # Compute gradient of soft-max ac w.r.t. x
        beta = 100.0
        val, grad_x = _soft_max_and_grad(x_norm, beta)
        
        # Compute gradient w.r.t. amplitude in frequency domain
        G = np.fft.rfft(grad_x, n=L_fft)
        grad_mag = np.zeros_like(mag)
        if L_fft > 1:
            grad_mag[1:] = (2.0 / L_fft) * np.real(np.conj(X[1:]) * G[1:])
        grad_mag[0] = np.real(G[0]) / L_fft
        
        # Gradient descent on amplitude
        step_size = 1.0 / (beta * np.max(mag) + 1e-6)
        mag = mag - step_size * grad_mag
        mag = np.maximum(mag, 0.01)  # Keep amplitudes positive
        mag[0] = np.maximum(mag[0], 0.01)
        
        # Re-normalize to keep sum(x) = 1
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_new = np.fft.irfft(spectrum, n=L_fft)[:N]
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            x_norm = x_new / s_new
        else:
            x_norm = x_new
    
    return x_norm

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x, 50.0)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate focused initial points
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        np.exp(1.0 * t),
        np.exp(-1.0 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        1.0 + 0.2 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(3 * np.pi * t),
    ]
    
    # Add spectral initializations with phase optimization
    for mode in range(4):
        for _ in range(2):
            if _check_deadline():
                break
            initial_points.append(_spectral_init_phase_optimized(rng, mode, iters=8))
    
    # Add a few random initializations
    for _ in range(5):
        if _check_deadline():
            break
        scale = rng.uniform(0.8, 1.2)
        initial_points.append(scale * rng.normal(1.0, 0.15, size=N))
    
    # L-BFGS optimization from each initial point
    betas = [50.0, 100.0, 200.0]
    
    candidates = []
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 200, 'gtol': 1e-9, 'ftol': 1e-11})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                
                candidates.append((result.fun, result.x))
                
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Sort candidates and polish the top few
    candidates.sort(key=lambda item: item[0])
    top_candidates = [x for _, x in candidates[:10]]
    
    # Amplitude optimization on top candidates
    for x_init in top_candidates:
        if _check_deadline():
            break
        try:
            x_amp = _amplitude_optimize(x_init, iters=15)
            val_amp = _objective_only(x_amp, 200.0)
            if val_amp < best_val:
                best_val = val_amp
                best_x = x_amp.copy()
        except Exception:
            pass
    
    # Peak-targeted polish on top candidates
    for x_init in top_candidates:
        if _check_deadline():
            break
        try:
            x_peak, val_peak = _peak_targeted_polish(x_init, max_iters=60, beta_start=100.0, beta_end=500.0)
            if val_peak < best_val:
                best_val = val_peak
                best_x = x_peak.copy()
        except Exception:
            pass
    
    # Final Polish with high beta using L-BFGS
    for x_init in top_candidates[:6]:
        if _check_deadline():
            break
        
        for beta in [300.0, 500.0, 800.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_init, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 400, 'gtol': 1e-11, 'ftol': 1e-13})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    # Final coordinate descent polish
    if not _check_deadline():
        step_size = 0.01
        beta_fast = 800.0
        no_improve = 0
        max_no_improve = 3000
        
        while no_improve < max_no_improve and not _check_deadline():
            # Try perturbing a small random subset of indices
            n_perturb = rng.integers(1, 5)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
            val = _objective_only(x_trial, beta_fast)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                step_size = max(0.001, step_size * 0.95)
                no_improve = 0
            else:
                no_improve += 1
                step_size = min(0.05, step_size * 1.05)
                
            if no_improve > 400:
                # If stuck, try a larger jump
                perturbation = np.zeros(N)
                indices = rng.choice(N, size=10, replace=False)
                perturbation[indices] = rng.normal(0, 0.05, size=10)
                x_trial = best_x + perturbation
                val = _objective_only(x_trial, beta_fast)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
                    no_improve = 0
                    step_size = 0.01
                else:
                    no_improve += 1

    return best_x
# EVOLVE-BLOCK-END
