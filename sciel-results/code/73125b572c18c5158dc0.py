"""Model-and-optimize: spectral phase retrieval + L-BFGS + coordinate descent + spectral amplitude polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Cross-correlation: corr[n] = sum_k grad_ac[k] * x[k-n] for 0<=k-n<N
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase retrieval.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_phase_optimize(x0, rng, max_iters=200, lr=0.01):
    """
    Optimize the phase of the spectrum while keeping magnitude fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        val, grad = _objective_and_grad(best_x, 50.0)
        
        G = np.fft.rfft(grad, n=L_fft)
        
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        
        phase -= lr * phase_grad
        
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=40, beta=50.0):
    """
    Optimize the amplitude spectrum while keeping phase fixed.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    
    best_amp = amp.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            amp = amp_trial.copy()
            best_amp = amp.copy()
    
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _peak_targeted_polish(x0, rng, max_iters=1200, step_size=0.01):
    """
    Identify dominant autoconvolution lags and apply targeted perturbations.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if it % 15 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        top_k = 10
        sorted_idx = np.argsort(-abs_ac)[:top_k]
        
        improved = False
        for k in sorted_idx:
            if _check_deadline():
                break
            
            n_lo = max(0, k - N + 1)
            n_hi = min(k, N - 1)
            
            if n_hi >= n_lo:
                idx_in_x = k - np.arange(n_lo, n_hi + 1)
                contrib_segment = 2.0 * best_x[idx_in_x]
                
                contrib = np.zeros(N)
                contrib[n_lo:n_hi+1] = contrib_segment
                
                norm_contrib = np.linalg.norm(contrib)
                if norm_contrib > 1e-10:
                    direction = -contrib / norm_contrib
                    
                    for scale_mult in [1.0, 0.5, 1.5, 2.0, 0.25]:
                        scale = step_size * scale_mult
                        x_trial = best_x + scale * direction
                        val = _objective_only(x_trial)
                        if val < best_val:
                            best_val = val
                            best_x = x_trial.copy()
                            improved = True
                            break
        
        if improved:
            step_size = max(0.0005, step_size * 0.92)
        else:
            step_size = min(0.1, step_size * 1.04)
        
        if it % 500 == 499:
            x_perturbed = best_x + rng.normal(0, 0.05, size=N)
            val_pert = _objective_only(x_perturbed)
            if val_pert < best_val:
                best_val = val_pert
                best_x = x_perturbed.copy()
                step_size = 0.02
    
    return best_x, best_val

def _epigraph_optimize(x0, rng, max_iters=300, alpha_start=10.0):
    """
    Epigraph formulation: minimize t + alpha * sum_k max(0, |C_k| - t)^2
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    alpha = alpha_start
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        s = np.sum(best_x)
        s2 = s * s
        
        t = m / s2
        
        violations = np.maximum(0, abs_ac / s2 - t)
        penalty = alpha * np.sum(violations ** 2)
        obj_val = t + penalty
        
        _, grad_soft = _soft_max_and_grad(best_x, 100.0)
        grad_x = grad_soft / s2 - 2.0 * (m/s2) * np.ones(N) / (s2 * s)
        
        lr = 0.01
        for _ in range(5):
            x_trial = best_x - lr * grad_x
            val_trial = _objective_only(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial.copy()
                break
            lr *= 0.5
        
        alpha *= 0.98
    
    return best_x, best_val

def _adaptive_beta_lbfgs(x0, rng, max_iters=150):
    """
    L-BFGS with adaptive beta scheduling based on convergence behavior.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    beta = 50.0
    no_improve = 0
    max_no_improve = 20
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 30})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                no_improve = 0
                beta = max(10.0, beta * 0.9)
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    beta = min(5000.0, beta * 1.5)
                    no_improve = 0
        except Exception:
            pass
    
    return best_x, best_val

def _multi_frequency_polish(x0, rng, max_iters=800):
    """
    Coordinate descent with multi-frequency sinusoidal perturbations.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    step_size = 0.02
    no_improve = 0
    t_arr = np.arange(N)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        r = rng.random()
        if r < 0.2:
            # Single point
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.4:
            # Small random block
            n_perturb = rng.integers(2, 8)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.6:
            # Low-frequency sinusoid
            freq = rng.integers(1, 5)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.8:
            # Multi-frequency combination
            freq1 = rng.integers(1, 4)
            freq2 = rng.integers(5, 12)
            phase1 = rng.uniform(0, 2 * np.pi)
            phase2 = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * (0.7 * np.sin(2 * np.pi * freq1 * t_arr / N + phase1) +
                                        0.3 * np.sin(2 * np.pi * freq2 * t_arr / N + phase2))
            x_trial = best_x + perturbation
        else:
            # Peak-targeted perturbation
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
            top_lag = np.argmax(abs_ac)
            n_lo = max(0, top_lag - N + 1)
            n_hi = min(top_lag, N - 1)
            if n_hi >= n_lo:
                n_perturb = rng.integers(2, min(10, n_hi - n_lo + 1))
                indices = rng.choice(np.arange(n_lo, n_hi + 1), size=n_perturb, replace=False)
                perturbation = np.zeros(N)
                perturbation[indices] = rng.normal(0, step_size * 1.5, size=n_perturb)
                x_trial = best_x + perturbation
            else:
                x_trial = best_x + rng.normal(0, step_size, size=N)
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0003, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.12, step_size * 1.03)
        
        if no_improve > 3000 and rng.random() < 0.05:
            x_restart = best_x + rng.normal(0, 0.08, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
                step_size = 0.02
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
        1.0 + 0.3 * np.sin(2 * np.pi * 5 * t + 0.5),
        1.0 + 0.4 * np.cos(2 * np.pi * 7 * t),
        1.0 + 0.2 * np.sin(2 * np.pi * 11 * t) + 0.2 * np.cos(2 * np.pi * 13 * t),
        np.maximum(0.2, 1.0 - 4.0 * np.abs(t - 0.5)),
        1.0 + 0.5 * np.sin(2 * np.pi * t) * np.sin(4 * np.pi * t),
        0.8 + 0.4 * np.cos(2 * np.pi * 2 * t + 1.0),
        np.exp(-3.0 * (t - 0.5)**2),
        1.0 + 0.3 * np.tanh(3.0 * (t - 0.5)),
    ]
    
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(8):
        scale = rng.uniform(0.4, 1.6)
        initial_points.append(scale * rng.normal(1.0, 0.25, size=N))
    
    for _ in range(4):
        x_sparse = np.ones(N)
        n_sparse = rng.integers(10, 50)
        idx = rng.choice(N, size=n_sparse, replace=False)
        x_sparse[idx] = rng.uniform(0.3, 1.5, size=n_sparse)
        initial_points.append(x_sparse)
    
    # Phase 1: Adaptive beta L-BFGS from multiple initial points
    for x0 in initial_points[:15]:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        x_opt, val_opt = _adaptive_beta_lbfgs(x_start, rng, max_iters=80)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=150, lr=0.015)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Amplitude optimization
    for _ in range(2):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=25)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=800, step_size=0.01)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Multi-frequency coordinate descent
    x_mf, val_mf = _multi_frequency_polish(best_x, rng, max_iters=600)
    if val_mf < best_val:
        best_val = val_mf
        best_x = x_mf.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final high-precision L-BFGS with very high beta
    for beta in [500.0, 2000.0, 5000.0, 10000.0]:
        if _check_deadline():
            break
        try:
            def final_obj_grad(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(final_obj_grad, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 150, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 25})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final coordinate descent refinement
    x_final, val_final = _multi_frequency_polish(best_x, rng, max_iters=400)
    if val_final < best_val:
        best_val = val_final
        best_x = x_final.copy()
    
    return best_x
# EVOLVE-BLOCK-END
