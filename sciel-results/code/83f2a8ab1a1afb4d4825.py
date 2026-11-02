"""Model-and-optimize: epigraph formulation + multi-restart L-BFGS + spectral polish + coordinate descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 148

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 2)

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
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    L_corr = 1
    while L_corr < 4 * N:
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

def _epigraph_objective_and_grad(z, t, beta):
    """
    Epigraph formulation: minimize t subject to |ac[k]| <= t * (sum x)^2 for all k.
    Variables: x (N), t (scalar)
    Objective: t + lambda * soft_max(|ac|/s^2 - t, beta)
    """
    x = z[:N]
    t_val = z[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N + 1)
    
    ac = _autoconv_fft(x)
    s2 = s * s
    abs_ac = np.abs(ac) / s2
    
    # soft_max of (abs_ac - t)
    shifted = abs_ac - t_val
    m = np.max(shifted)
    if m < -1e-8:
        soft_pen = 0.0
        weights = np.zeros(2*N-1)
        signs = np.zeros(2*N-1)
    else:
        exps = np.exp(beta * (shifted - m))
        log_sum = np.log(np.sum(exps))
        soft_pen = m + log_sum / beta
        weights = exps / np.sum(exps)
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
    
    obj_val = t_val + 100.0 * soft_pen
    
    # Gradient w.r.t. x: need d(soft_pen)/dx[n]
    # soft_pen = soft_max_k (|ac[k]|/s^2 - t)
    # d(soft_pen)/dx[n] = sum_k weights[k] * sign(ac[k]) * d(ac[k]/s^2)/dx[n]
    # d(ac[k]/s^2)/dx[n] = sign(ac[k])*d(ac[k])/dx[n]/s^2 - 2*ac[k]*sign(ac[k])/s^3
    # d(ac[k])/dx[n] = 2*sum_i x[i]*x[i+k-n] (cross-correlation)
    
    if m < -1e-8:
        grad_x = np.zeros(N)
    else:
        # Compute cross-correlation: corr[n] = sum_k weights[k]*sign(ac[k]) * d(ac[k])/dx[n]
        # d(ac[k])/dx[n] = 2 * sum_i x[i] * x[i + k - n]
        # This is a convolution-like operation
        L_corr = 1
        while L_corr < 4 * N:
            L_corr *= 2
        
        # g[k] = weights[k] * sign(ac[k]) / s^2
        g = weights * signs / s2
        
        # h[n] = -2 * ac[k] * sign(ac[k]) / s^3 * weights[k] -> this couples k and n
        # For simplicity, use the dominant term: g[k] * d(ac[k])/dx[n]
        g_padded = np.zeros(L_corr)
        g_padded[:2*N-1] = g
        
        X_f = np.fft.rfft(x, n=L_corr)
        G_f = np.fft.rfft(g_padded, n=L_corr)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
        grad_x = 100.0 * 2.0 * corr[:N]
        
        # Add correction for the -2*ac*sign/s^3 term (approximate)
        # This is O(1/s^3) so smaller; include it for accuracy
        h = weights * signs * ac / (s2 * s)  # = weights*sign*ac/s^3
        h_padded = np.zeros(L_corr)
        h_padded[:2*N-1] = h
        H_f = np.fft.rfft(h_padded, n=L_corr)
        corr2 = np.fft.irfft(H_f * np.conj(X_f), n=L_corr)
        grad_x -= 100.0 * 2.0 * corr2[:N]
    
    # Gradient w.r.t. t: d(soft_max(shifted))/dt = -1 always
    grad_t = 1.0 - 100.0
    
    grad = np.concatenate([grad_x, [grad_t]])
    
    return obj_val, grad

def _peak_targeted_polish(x0, rng, max_iters=2000, step_size=0.05):
    """
    Targeted coordinate descent that specifically attacks the dominant
    autoconvolution lags. Identifies top K lags and applies targeted
    perturbations to reduce them.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step = step_size
    no_improve = 0
    K = 10  # number of dominant lags to track
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s2 = s * s
        abs_ac = np.abs(ac) / s2
        
        # Find top K lags
        top_k = np.argsort(abs_ac)[-K:]
        
        # For each top lag, try to reduce it
        for lag in top_k:
            if _check_deadline():
                break
            
            # Determine which x indices affect ac[lag]
            # ac[lag] = sum_i x[i] * x[i+lag] for lag >= 0
            #          = sum_i x[i] * x[i-lag] for lag < 0
            # Perturb x to reduce |ac[lag]|
            
            # Compute gradient of |ac[lag]| w.r.t. x
            # d(ac[lag])/dx[n] = 2 * sum_i x[i] * delta(i + lag - n, i)
            # For lag >= 0: d(ac[lag])/dx[n] = 2*x[n-lag] if 0<=n-lag<N else 0
            #                 + 2*x[n+lag] if 0<=n+lag<N else 0 (from symmetry)
            
            sign_ac = np.sign(ac[lag])
            
            # Gradient of |ac[lag]|/s^2 w.r.t. x
            grad_lag = np.zeros(N)
            if lag >= 0:
                # ac[lag] = sum_{i=0}^{N-1-lag} x[i]*x[i+lag]
                # d(ac[lag])/dx[n] = x[n-lag] (if 0<=n-lag<N-lag) + x[n+lag] (if 0<=n+lag<N)
                for n in range(N):
                    g = 0.0
                    if 0 <= n - lag < N - lag:
                        g += x0[n - lag]  # Use best_x for stability
                        g += best_x[n - lag]
                    if 0 <= n + lag < N:
                        g += best_x[n + lag]
                    grad_lag[n] = sign_ac * g / s2
            else:
                lag_pos = -lag
                for n in range(N):
                    g = 0.0
                    if 0 <= n - lag_pos < N - lag_pos:
                        g += best_x[n - lag_pos]
                    if 0 <= n + lag_pos < N:
                        g += best_x[n + lag_pos]
                    grad_lag[n] = sign_ac * g / s2
            
            # Apply small step in negative gradient direction
            x_trial = best_x - step * grad_lag
            
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                no_improve = 0
            else:
                no_improve += 1
        
        if no_improve > 500:
            step = max(0.005, step * 0.95)
            no_improve = 0
        
        if it % 100 == 0:
            step = min(0.1, step * 1.02)
    
    return best_x, best_val

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase retrieval."""
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

def _spectral_phase_optimize(x0, rng, max_iters=150, lr=0.01):
    """Optimize the phase of the spectrum while keeping magnitude fixed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
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

def _amplitude_optimize(x0, rng, iters=30):
    """Optimize the amplitude spectrum while keeping phase fixed."""
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

def _coordinate_descent(x0, rng, max_iters=15000, initial_step=0.02):
    """Coordinate descent with structured perturbations."""
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step_size = initial_step
    no_improve = 0
    
    for _ in range(max_iters):
        if _check_deadline():
            break
        
        r = rng.random()
        if r < 0.25:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.5:
            n_perturb = rng.integers(2, 8)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.7:
            freq = rng.integers(1, 6)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(N)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.85:
            n_perturb = rng.integers(5, 20)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size * 2, size=n_perturb)
            x_trial = best_x + perturbation
        else:
            # Sign-flip on a random subset
            n_flip = rng.integers(3, 15)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] = -x_trial[indices]
            # Scale back toward original magnitude
            x_trial[indices] *= 0.7
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.05, step_size * 1.015)
        
        if no_improve > 3000 and rng.random() < 0.02:
            x_restart = best_x + rng.normal(0, 0.05, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
    
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
    ]
    
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(10):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Phase 1: Multi-restart L-BFGS with soft-max (reduced calls)
    betas = [60.0, 120.0]
    
    for x0 in initial_points[:15]:  # Limit initial points to save time
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
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 30})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Peak-targeted polish (new, from inspirations)
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=3000, step_size=0.04)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=120, lr=0.012)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Amplitude optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=20)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-beta L-BFGS polish
    for beta in [400.0, 800.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 120, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 25})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent refinement
    best_x, best_val = _coordinate_descent(best_x, rng, max_iters=15000, initial_step=0.015)
    
    # Phase 7: Peak-targeted polish again after coordinate descent
    if not _check_deadline():
        x_peak2, val_peak2 = _peak_targeted_polish(best_x, rng, max_iters=2000, step_size=0.03)
        if val_peak2 < best_val:
            best_val = val_peak2
            best_x = x_peak2.copy()
    
    # Phase 8: Final high-beta polish
    if not _check_deadline():
        for beta in [600.0]:
            if _check_deadline():
                break
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 80, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 20})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    return best_x
# EVOLVE-BLOCK-END
