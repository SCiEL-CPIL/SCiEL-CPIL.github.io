"""Model-and-optimize: spectral phase retrieval + L-BFGS + coordinate descent + spectral amplitude polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

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
    
    The autoconvolution ac[k] = sum_i x[i]*x[i+k] for k >= 0
    and ac[k] = ac[-k] for k < 0 (even symmetry).
    
    For the full linear autoconvolution of length 2N-1:
    ac_full[m] = sum_{i} x[i] * x[i + m - (N-1)] where valid indices
    
    Gradient of ac_full[m] w.r.t. x[j]:
    d(ac_full[m])/dx[j] = x[j + m - (N-1)] + x[j - m + (N-1)]
    (with appropriate boundary handling)
    
    Using FFT: if G is the gradient weights on ac_full, then
    grad_x[j] = sum_m G[m] * (x[j+m-(N-1)] + x[j-m+(N-1)])
    This can be computed as a convolution/correlation.
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
    grad_ac = weights * signs
    
    # Gradient w.r.t. x via FFT-based correlation
    g = grad_ac  # length 2N-1
    
    # Use FFT-based approach: grad_x[j] = sum_m g[m] * (x[j+m-(N-1)] + x[j-m+(N-1)])
    # This is equivalent to: convolve g with x in a specific way
    # Let's use the direct formula: for each j, grad_x[j] = sum over valid m of g[m]*x[j+m-(N-1)] + g[m]*x[j-m+(N-1)]
    
    # Efficient computation using FFT
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g, n=L_corr)
    
    # First term: sum_m g[m] * x[j+m-(N-1)] = (g * x_shifted)[j]
    # where x_shifted[k] = x[k-(N-1)] (with zeros outside [0,N-1])
    # This is cross-correlation: corr1[j] = sum_m g[m] * x[j+m-(N-1)]
    # = sum_k g[k-(j-(N-1))] * x[k] = (g_shifted * x)[j]
    
    # Actually, let's think more carefully.
    # ac[m] = sum_i x[i] * x[i+m-(N-1)] for m in [0, 2N-2]
    # d(ac[m])/dx[j] = x[j+m-(N-1)] + x[j-m+(N-1)] (when indices valid)
    
    # So grad_x[j] = sum_m g[m] * (x[j+m-(N-1)] + x[j-m+(N-1)])
    # Term 1: sum_m g[m] * x[j+m-(N-1)]
    # Term 2: sum_m g[m] * x[j-m+(N-1)]
    
    # For term 1: let k = j+m-(N-1), so m = k-j+(N-1)
    # sum_k g[k-j+(N-1)] * x[k] = (g_rev_shifted * x)[j]
    
    # Let's just do it with the cumsum approach which was working:
    cumsum_g = np.concatenate([[0], np.cumsum(g)])
    idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
    idx_start = np.arange(N)
    window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
    term2 = x * window_sums
    
    corr = np.fft.irfft(X_f * np.conj(G_f), n=L_corr)
    term1 = corr[:N]
    
    grad_x = term1 + term2
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
    
    freqs = np.fft.rfftfreq(L_fft)
    n_freq = L_fft // 2 + 1
    
    if mode == 0:
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        cutoff = 0.4
        mag = np.where(np.abs(freqs) < cutoff, 1.0, 0.1)
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)
    
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    s = np.sum(x)
    if s < 0:
        x = -x
    return x

def _spectral_phase_optimize(x0, rng, iters=50, step_scale=0.1):
    """
    Optimize the phase spectrum while keeping amplitude fixed.
    Different parameterization from amplitude optimization.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x0, n=L_fft)
    amp = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    
    best_phase = phase.copy()
    best_val = _objective_only(x0)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        phase_trial = phase + step_scale * rng.normal(0, 1, size=n_freq)
        phase_trial[0] = 0
        
        spectrum = amp * np.exp(1j * phase_trial)
        spectrum[0] = amp[0]
        x_trial = np.fft.irfft(spectrum, n=L_fft)[:N]
        
        s = np.sum(x_trial)
        if s < 0:
            x_trial = -x_trial
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            phase = phase_trial.copy()
            best_phase = phase.copy()
    
    spectrum = best_phase * np.exp(1j * best_phase)
    spectrum[0] = amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _amplitude_optimize(x0, rng, iters=30, beta=50.0):
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

def _peak_targeted_polish(x0, rng, max_iters=500):
    """
    Identify dominant autoconvolution lags and apply targeted perturbations
    to reduce them. This is a local search that focuses on the peaks.
    """
    x = x0.copy()
    best_val = _objective_only(x)
    step_size = 0.05
    no_improve = 0
    max_no_improve = 100
    
    for it in range(max_iters):
        if _check_deadline():
            break
        if no_improve > max_no_improve:
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        # Find top peaks (by absolute value)
        top_k = 10
        peak_indices = np.argsort(abs_ac)[-top_k:]
        
        # For each peak, try to reduce it by perturbing x
        improved = False
        for lag in peak_indices:
            if _check_deadline():
                break
            
            # The peak at lag m means ac[m] is large.
            # ac[m] = sum_i x[i]*x[i+m-(N-1)]
            # To reduce |ac[m]|, we can try to make contributions cancel.
            
            # Try small perturbations in regions that affect this lag
            # Perturb x in a window around the center
            center = N // 2
            window_size = rng.integers(5, 20)
            start = max(0, center - window_size)
            end = min(N, center + window_size)
            
            n_perturb = rng.integers(2, 8)
            indices = rng.choice(np.arange(start, end), size=min(n_perturb, end-start), replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=len(indices))
            
            x_trial = x + perturbation
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                x = x_trial.copy()
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
            step_size = max(0.005, step_size * 0.95)
    
    return x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
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
    
    # Add spectral initializations
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    # Add random initializations
    for _ in range(8):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # L-BFGS optimization from each initial point
    betas = [30.0, 80.0]
    
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
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Spectral phase optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_phase, val_phase = _spectral_phase_optimize(best_x, rng, iters=50, step_scale=0.1)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Amplitude optimization
    for _ in range(5):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=40, beta=50.0)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=500)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Polish with higher beta
    for beta in [150.0, 400.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-11, 'ftol': 1e-13})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Final peak-targeted polish after L-BFGS
    x_peak2, val_peak2 = _peak_targeted_polish(best_x, rng, max_iters=300)
    if val_peak2 < best_val:
        best_val = val_peak2
        best_x = x_peak2.copy()
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement
    step_size = 0.01
    no_improve = 0
    max_no_improve = 10000
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = rng.integers(1, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.05, step_size * 1.005)
    
    return best_x
# EVOLVE-BLOCK-END
