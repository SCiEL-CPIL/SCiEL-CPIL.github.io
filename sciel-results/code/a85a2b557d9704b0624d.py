"""Alternating minimization with soft spectral projection for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 145
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

def _soft_max_and_grad(x, beta):
    """Compute soft-max surrogate for max|ac| and its gradient."""
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
    cumsum_g = np.concatenate([[0], np.cumsum(g)])
    idx_end = np.minimum(np.arange(N) + N - 1, 2*N-2) + 1
    idx_start = np.arange(N)
    window_sums = cumsum_g[idx_end] - cumsum_g[idx_start]
    term2 = x * window_sums
    
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g, n=L_corr)
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

def _flat_top_mag(L_fft, beta=6.0):
    """Generate flat-top magnitude spectrum."""
    freqs = np.fft.rfftfreq(L_fft)
    # Flat-top window: 1 - 0.5*cos(2*pi*f) + beta*cos(4*pi*f) - 0.5*beta*cos(6*pi*f)
    mag = (1.0 - 0.5 * np.cos(2 * np.pi * freqs) 
             + beta * np.cos(4 * np.pi * freqs) 
             - 0.5 * beta * np.cos(6 * np.pi * freqs))
    mag = np.maximum(mag, 0.01)
    return mag

def _soft_project_to_mag(x, target_mag, L_fft, mu=100.0):
    """
    Soft project x onto the set of signals with magnitude close to target_mag.
    Uses a penalty approach: minimize ||x||^2 + mu * sum((|X| - target_mag)^2)
    """
    X = np.fft.rfft(x, n=L_fft)
    current_mag = np.abs(X)
    phase = np.angle(X)
    phase[0] = 0.0
    
    # Adjust magnitude toward target
    # x_projected has X_projected = target_mag * exp(i*phase)
    # But we do a soft update: blend current with target
    X_new = target_mag * np.exp(1j * phase)
    X_new[0] = target_mag[0]
    
    x_new = np.fft.irfft(X_new, n=L_fft)[:N]
    
    # Soft: blend original and projected
    x_soft = x + 0.5 * (x_new - x)
    
    return x_soft

def _alternating_minimize(x0, target_mag, L_fft, max_iters=50, lr=0.01, beta=50.0, proj_strength=0.5):
    """
    Alternating minimization:
    1. Update x to minimize soft-max objective (gradient step)
    2. Soft project x toward fixed magnitude target_mag
    
    This decouples phase optimization from amplitude shaping.
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _objective(x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Step 1: Gradient descent on soft-max objective
        val, grad = _objective_and_grad(x, beta)
        
        # Adaptive learning rate based on gradient norm
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            step = 0.0
        else:
            step = lr / (gnorm + 1e-10)
        
        x = x - step * grad
        
        # Step 2: Soft spectral projection
        x = (1.0 - proj_strength) * x + proj_strength * _soft_project_to_mag(x, target_mag, L_fft)
        
        # Ensure positive sum
        s = np.sum(x)
        if s < 0:
            x = -x
        
        # Track best
        val_curr = _objective(x)
        if val_curr < best_val:
            best_val = val_curr
            best_x = x.copy()
        
        # Decay learning rate
        lr *= 0.97
        # Increase projection strength over time
        proj_strength = min(0.8, 0.3 + 0.01 * it)
    
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

def _coordinate_descent_polish(x, max_iters=100):
    """Fast coordinate descent to polish the solution."""
    best_x = x.copy()
    best_val = _objective(best_x)
    
    deltas = [0.05, -0.05, 0.1, -0.1, 0.02, -0.02]
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            for d in deltas:
                trial = best_x.copy()
                trial[j] += d
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Generate target magnitude spectra for different flat-top variants
    target_mags = [
        _flat_top_mag(L_fft, beta=4.0),
        _flat_top_mag(L_fft, beta=6.0),
        _flat_top_mag(L_fft, beta=8.0),
        np.ones(L_fft // 2 + 1),  # Flat magnitude
    ]
    
    # Add some structured targets
    freqs = np.fft.rfftfreq(L_fft)
    target_mags.append(np.exp(-5 * (freqs * L_fft / 2) ** 2))  # Gaussian
    target_mags.append(np.where(np.abs(freqs) < 0.3, 1.0, 0.2))  # Low-pass
    
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
    ]
    
    # Add spectral inits
    for mode in range(5):
        for _ in range(2):
            if _check_deadline():
                break
            x_spec = _spectral_init(rng, mode)
            initial_points.append(x_spec)
    
    # Run alternating minimization with various targets and initial points
    # Time budget: ~130s total, allocate ~8s per (init, target) pair
    max_pairs = 15
    
    pair_count = 0
    for x0 in initial_points:
        if _check_deadline() or pair_count >= max_pairs:
            break
        
        for tmag in target_mags:
            if _check_deadline() or pair_count >= max_pairs:
                break
            
            try:
                x_alt, val_alt = _alternating_minimize(
                    x0, tmag, L_fft, 
                    max_iters=40, lr=0.02, beta=50.0, proj_strength=0.4
                )
                if val_alt < best_val:
                    best_val = val_alt
                    best_x = x_alt.copy()
                pair_count += 1
            except Exception:
                pass
    
    # Final polish with coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=80)
    
    return best_x
# EVOLVE-BLOCK-END
