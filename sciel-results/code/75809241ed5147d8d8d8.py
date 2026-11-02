"""Model-and-optimize: Chebyshev spectral envelope + phase retrieval + L-BFGS + coordinate descent."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 148

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
    Returns soft_val, grad_x, and the adversary weights.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x, weights

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft, _ = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

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

def _chebyshev_envelope_optimize(x0, rng, max_iters=300):
    """
    Chebyshev-like optimization of the spectral envelope.
    We optimize the magnitude of the FFT (envelope) to minimize the L_inf norm
    of the autoconvolution, using a Remez-like exchange strategy on the
    frequency grid, then recover x via phase retrieval.
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
    mag[0] = np.real(X[0])  # Keep DC real and positive
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    best_mag = mag.copy()
    
    # Remez-like: track the "extremal" frequencies where the autoconvolution
    # peak is dominated. We adjust the envelope to flatten the response.
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Reconstruct current x
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_curr = np.fft.irfft(spectrum, n=L_fft)[:N]
        s = np.sum(x_curr)
        if s < 0:
            x_curr = -x_curr
            phase = -phase
            phase[0] = 0
            mag[0] = -mag[0]
        
        val_curr = _objective_only(x_curr)
        
        # Compute the autoconvolution in frequency domain
        X_curr = np.fft.rfft(x_curr, n=L_fft)
        AC_spec = X_curr * X_curr  # Power spectrum
        ac_time = np.fft.irfft(AC_spec, n=L_fft)[:2*N-1]
        abs_ac = np.abs(ac_time)
        peak_idx = np.argmax(abs_ac)
        
        # The peak in time domain corresponds to a specific structure in freq.
        # We use a soft-max surrogate on the time-domain autoconvolution
        # and back-propagate to the magnitude spectrum.
        
        beta = 50.0
        m = np.max(abs_ac)
        exps = np.exp(beta * (abs_ac - m))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        signs = np.sign(ac_time)
        signs[ac_time == 0] = 1.0
        grad_ac = weights * signs
        
        # Gradient w.r.t. x: cross-correlation
        L_corr = 1
        while L_corr < 3 * N:
            L_corr *= 2
        g_padded = np.zeros(L_corr)
        g_padded[:2*N-1] = grad_ac
        X_f = np.fft.rfft(x_curr, n=L_corr)
        G_f = np.fft.rfft(g_padded, n=L_corr)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
        grad_x = 2.0 * corr[:N]
        
        # Gradient w.r.t. magnitude spectrum
        G_x = np.fft.rfft(grad_x, n=L_fft)
        # d/dmag[|X|] = Re(X * conj(G_x)) / |X|
        mag_grad = np.real(X_curr * np.conj(G_x)) / (np.abs(X_curr) + 1e-10)
        mag_grad[0] = 0  # DC is fixed
        
        # Take a step in magnitude
        lr = 0.01 * (1.0 / (1.0 + 0.01 * it))
        mag_trial = mag - lr * mag_grad
        mag_trial = np.maximum(0.01, mag_trial)
        
        # Reconstruct
        spectrum_trial = mag_trial * np.exp(1j * phase)
        spectrum_trial[0] = mag_trial[0]
        x_trial = np.fft.irfft(spectrum_trial, n=L_fft)[:N]
        s_trial = np.sum(x_trial)
        if s_trial < 0:
            x_trial = -x_trial
        
        val_trial = _objective_only(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            best_mag = mag_trial.copy()
            mag = mag_trial.copy()
        else:
            # If no improvement, try a small random perturbation of the envelope
            if rng.random() < 0.3:
                perturb_idx = rng.integers(1, n_freq)
                mag[perturb_idx] *= (1 + 0.05 * rng.normal(0, 1))
                mag[perturb_idx] = max(0.01, mag[perturb_idx])
    
    return best_x, best_val, best_mag

def _spectral_phase_optimize(x0, rng, max_iters=200, lr=0.01):
    """Optimize the phase of the spectrum while keeping magnitude fixed."""
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
            best_amp = amp_trial.copy()
    
    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _game_based_polish(x0, rng, max_iters=300, beta=100.0, lr=0.01):
    """
    Game-theoretic polish: Uses the soft-max weights (adversary strategy)
    to compute a weighted gradient and take steps.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        s = np.sum(best_x)
        if abs(s) < 1e-10:
            break
        s2 = s * s
        
        soft_val, grad_soft, weights = _soft_max_and_grad(best_x, beta)
        
        grad_obj = grad_soft / s2 - 2.0 * soft_val * np.ones(N) / (s2 * s)
        
        x_trial = best_x - lr * grad_obj
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            lr *= 0.99
        else:
            lr *= 0.9
            if rng.random() < 0.1:
                best_x += rng.normal(0, 0.01, size=N)
    
    return best_x, best_val

def _peak_targeted_polish(x0, rng, max_iters=1200, step_size=0.01):
    """
    Identify dominant autoconvolution lags and apply targeted perturbations
    to reduce the peak absolute value.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if it % 15 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        top_k = 10
        sorted_idx = np.argsort(abs_ac)[::-1][:top_k]
        
        # For each peak lag, try to reduce it by perturbing x
        for k in sorted_idx:
            if _check_deadline():
                break
            
            # d(ac[k])/dx[n] = 2*x[k-n] (for valid n)
            # We want to reduce |ac[k]|, so move x in the direction that
            # reduces ac[k] * sign(ac[k])
            sign_k = np.sign(ac[k])
            if sign_k == 0:
                continue
            
            # Perturb x at positions that most affect lag k
            # x[k-n] contributes to ac[k]
            n_candidates = np.arange(N)
            contributions = 2.0 * np.roll(x0, -k)[:N]  # approximate
            # Actually, let's just do a small random perturbation weighted by x
            perturbation = step_size * sign_k * rng.normal(0, 1, size=N) * (1 + 0.5 * np.abs(best_x))
            x_trial = best_x + perturbation * 0.1  # small step
            
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
    
    return best_x, best_val

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
            n_flip = rng.integers(3, 15)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] = -x_trial[indices]
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
    
    # Phase 1: Multi-restart L-BFGS with dynamic beta
    betas = [30.0, 60.0, 120.0]
    
    for x0 in initial_points[:15]:
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
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 30})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Chebyshev spectral envelope optimization (NEW)
    # This is the key new component: optimize the spectral envelope
    # to minimize the L_inf norm of the autoconvolution
    x_cheb, val_cheb, _ = _chebyshev_envelope_optimize(best_x, rng, max_iters=200)
    if val_cheb < best_val:
        best_val = val_cheb
        best_x = x_cheb.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Game-theoretic polish
    x_game, val_game = _game_based_polish(best_x, rng, max_iters=200, beta=100.0, lr=0.02)
    if val_game < best_val:
        best_val = val_game
        best_x = x_game.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=100, lr=0.012)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Amplitude optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=20)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: High-beta L-BFGS polish
    for beta in [200.0, 400.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 25})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=800, step_size=0.01)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Coordinate descent refinement
    best_x, best_val = _coordinate_descent(best_x, rng, max_iters=10000, initial_step=0.015)
    
    # Phase 9: Final Game-based polish with high beta
    if not _check_deadline():
        x_game2, val_game2 = _game_based_polish(best_x, rng, max_iters=200, beta=500.0, lr=0.01)
        if val_game2 < best_val:
            best_val = val_game2
            best_x = x_game2.copy()
    
    # Phase 10: Final high-beta L-BFGS
    if not _check_deadline():
        try:
            def obj_and_grad_func(x, _beta=1000.0):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 20})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
