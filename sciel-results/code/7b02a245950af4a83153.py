"""Structural shift: spectral-domain Chebyshev moment-matching with adaptive L-BFGS polish."""
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
    
    # Compute grad_x via FFT correlation
    # Center grad_ac: index m from 0 to 2N-2, center at N-1
    h = np.zeros(2 * N - 1)
    h[N-1 : 2*N-1] = grad_ac[0 : N]
    h[0 : N-1] = grad_ac[2*N-2 : N-1 : -1]
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    X_f = np.fft.rfft(x, n=L_corr)
    H_f = np.fft.rfft(h, n=L_corr)
    
    corr1 = np.fft.irfft(np.conj(H_f) * X_f, n=L_corr)[:N]
    
    h_rev = h[::-1]
    H_rev_f = np.fft.rfft(h_rev, n=L_corr)
    corr2 = np.fft.irfft(np.conj(H_rev_f) * X_f, n=L_corr)[:N]
    
    grad_x = corr1 + corr2
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

def _spectral_chebyshev_init(rng, num_cheb=15):
    """
    Generate initial x via Chebyshev moment matching in spectral domain.
    
    Key insight: The autoconvolution ac[k] relates to the power spectrum via
    ac[k] = sum_j x[j] x[j+k]. In the frequency domain, the power spectrum
    |X(f)|^2 determines the autocorrelation structure.
    
    We parameterize the amplitude spectrum using Chebyshev polynomials to
    enforce smoothness and exploit the structure of the problem.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Frequency axis in [0, 0.5]
    freqs = np.linspace(0, 0.5, n_freq)
    
    # Chebyshev basis: T_k(2*f) for k=0..num_cheb-1
    # Map to [-1, 1] for standard Chebyshev
    t = 2 * freqs - 1  # in [-1, 0]
    
    # Generate random Chebyshev coefficients with decay
    coeffs = np.zeros(num_cheb)
    for k in range(num_cheb):
        # Decay with frequency to ensure low-pass character
        decay = np.exp(-0.3 * k)
        coeffs[k] = rng.normal(0, decay)
    
    # Evaluate Chebyshev polynomial
    mag = np.zeros(n_freq)
    for k in range(num_cheb):
        # T_k(x) = cos(k * arccos(x))
        angles = k * np.arccos(np.clip(t, -1, 1))
        mag += coeffs[k] * np.cos(angles)
    
    # Ensure positive magnitude
    mag = np.abs(mag) + 0.1
    # Normalize
    mag = mag / np.max(mag)
    
    # Random phase
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0  # DC component real
    
    # Build spectrum
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]  # Keep DC real and positive
    
    # Inverse FFT to get time domain
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    # Normalize so sum is positive
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _spectral_fourier_init(rng, num_fourier=12):
    """
    Generate initial x via Fourier series parameterization of amplitude.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    freqs = np.linspace(0, 0.5, n_freq)
    
    # Fourier series: mag(f) = sum_k (a_k cos(2*pi*k*f) + b_k sin(2*pi*k*f))
    a = np.zeros(num_fourier)
    b = np.zeros(num_fourier)
    for k in range(num_fourier):
        decay = np.exp(-0.2 * k)
        a[k] = rng.normal(0, decay)
        b[k] = rng.normal(0, decay)
    
    mag = np.zeros(n_freq)
    for k in range(num_fourier):
        mag += a[k] * np.cos(2 * np.pi * k * freqs)
        mag += b[k] * np.sin(2 * np.pi * k * freqs)
    
    mag = np.abs(mag) + 0.1
    mag = mag / np.max(mag)
    
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

def _spectral_gaussian_init(rng):
    """
    Generate initial x via Gaussian-tapered spectral amplitude with random phase.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    freqs = np.linspace(0, 0.5, n_freq)
    
    # Gaussian taper with random center and width
    center = rng.uniform(0.05, 0.45)
    width = rng.uniform(0.05, 0.25)
    mag = np.exp(-((freqs - center) ** 2) / (2 * width ** 2))
    mag = mag / np.max(mag)
    
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

def _spectral_flat_init(rng):
    """
    Generate initial x via flat-top spectral amplitude with random phase.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    freqs = np.linspace(0, 0.5, n_freq)
    
    # Flat-top with slight taper at edges
    taper = np.minimum(1.0, freqs / 0.02) * np.minimum(1.0, (0.5 - freqs) / 0.02)
    mag = taper
    mag = mag / (np.max(mag) + 1e-10)
    
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

def _spectral_peak_init(rng):
    """
    Generate initial x with peaks in the amplitude spectrum at random frequencies.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    freqs = np.linspace(0, 0.5, n_freq)
    
    # Start with low background
    mag = 0.1 * np.ones(n_freq)
    
    # Add 3-6 peaks at random frequencies
    num_peaks = rng.integers(3, 7)
    for _ in range(num_peaks):
        peak_freq = rng.uniform(0.02, 0.48)
        peak_width = rng.uniform(0.01, 0.08)
        peak_height = rng.uniform(0.3, 1.0)
        mag += peak_height * np.exp(-((freqs - peak_freq) ** 2) / (2 * peak_width ** 2))
    
    mag = mag / np.max(mag)
    
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

def _spectral_phase_optimize(x0, rng, iters=20):
    """
    Optimize the phase spectrum while keeping amplitude fixed.
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
        
        # Perturb phases
        phase_trial = phase + rng.normal(0, 0.1, size=n_freq)
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
    
    spectrum = amp * np.exp(1j * best_phase)
    spectrum[0] = amp[0]
    x_best = np.fft.irfft(spectrum, n=L_fft)[:N]
    s = np.sum(x_best)
    if s < 0:
        x_best = -x_best
    
    return x_best, best_val

def _amplitude_optimize(x0, rng, iters=30):
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

def _peak_targeted_polish(x, rng, num_lags=50):
    """
    Targeted polish: identify lags with high |ac| and perturb x to reduce them.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    s = np.sum(x)
    
    # Find top lags by absolute autoconvolution
    top_lags = np.argsort(abs_ac)[::-1][:num_lags]
    
    best_x = x.copy()
    best_val = _objective_only(x)
    
    for lag in top_lags:
        if _check_deadline():
            break
        
        # Try to reduce |ac[lag]| by perturbing x
        # ac[lag] = sum_j x[j] x[j+lag] (for lag >= 0)
        # Gradient of ac[lag] w.r.t. x[j] = x[j+lag] + x[j-lag] (with bounds)
        
        # Simple approach: try small perturbations that reduce the peak
        for _ in range(5):
            if _check_deadline():
                break
            
            # Random perturbation weighted by the gradient structure
            pert = np.zeros(N)
            n_pert = rng.integers(3, 15)
            indices = rng.choice(N, size=n_pert, replace=False)
            pert[indices] = rng.normal(0, 0.01, size=n_pert)
            
            x_trial = best_x + pert
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
    
    return best_x, best_val

def _structured_perturb(x, rng):
    """
    Generate structured perturbations based on spectral features.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    X = np.fft.rfft(x, n=L_fft)
    amp = np.abs(X)
    phase = np.angle(X)
    
    # Perturb amplitudes in a structured way
    freqs = np.linspace(0, 0.5, n_freq)
    
    # Low-frequency perturbation
    low_mask = freqs < 0.1
    amp_low = amp.copy()
    amp_low[low_mask] *= (1 + 0.1 * rng.normal(0, 1, size=np.sum(low_mask)))
    amp_low = np.maximum(0.01, amp_low)
    
    # Mid-frequency perturbation
    mid_mask = (freqs >= 0.1) & (freqs < 0.3)
    amp_mid = amp.copy()
    amp_mid[mid_mask] *= (1 + 0.15 * rng.normal(0, 1, size=np.sum(mid_mask)))
    amp_mid = np.maximum(0.01, amp_mid)
    
    # High-frequency perturbation
    high_mask = freqs >= 0.3
    amp_high = amp.copy()
    amp_high[high_mask] *= (1 + 0.2 * rng.normal(0, 1, size=np.sum(high_mask)))
    amp_high = np.maximum(0.01, amp_high)
    
    # Phase perturbation
    phase_pert = phase + rng.normal(0, 0.05, size=n_freq)
    phase_pert[0] = 0
    
    candidates = []
    for amp_c in [amp_low, amp_mid, amp_high]:
        spectrum = amp_c * np.exp(1j * phase_pert)
        spectrum[0] = amp_c[0]
        x_c = np.fft.irfft(spectrum, n=L_fft)[:N]
        s = np.sum(x_c)
        if s < 0:
            x_c = -x_c
        candidates.append(x_c)
    
    return candidates

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points using new spectral methods
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
        1.0 + 0.4 * np.sin(2 * np.pi * 5 * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 7 * t),
        np.maximum(0.2, 1.0 - 4.0 * np.abs(t - 0.5)),
        1.0 + 0.6 * np.sin(2 * np.pi * 2 * t) * np.sin(2 * np.pi * 3 * t),
    ]
    
    # Add new spectral initializations
    for _ in range(8):
        initial_points.append(_spectral_chebyshev_init(rng))
    for _ in range(8):
        initial_points.append(_spectral_fourier_init(rng))
    for _ in range(8):
        initial_points.append(_spectral_gaussian_init(rng))
    for _ in range(8):
        initial_points.append(_spectral_flat_init(rng))
    for _ in range(8):
        initial_points.append(_spectral_peak_init(rng))
    
    # Add random initializations
    for _ in range(10):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Phase 1: Exploration with lower beta
    betas_explore = [20.0, 40.0]
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.02, size=N)
        
        for beta in betas_explore:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral phase optimization from best solution
    for _ in range(4):
        if _check_deadline():
            break
        x_phase, val_phase = _spectral_phase_optimize(best_x, rng, iters=25)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Amplitude optimization
    for _ in range(6):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=40)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Peak-targeted polish
    for _ in range(4):
        if _check_deadline():
            break
        x_peak, val_peak = _peak_targeted_polish(best_x, rng, num_lags=80)
        if val_peak < best_val:
            best_val = val_peak
            best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Exploitation with higher beta from best solution
    betas_exploit = [80.0, 150.0]
    
    for beta in betas_exploit:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Structured perturbations
    for _ in range(20):
        if _check_deadline():
            break
        candidates = _structured_perturb(best_x, rng)
        for x_c in candidates:
            val_c = _objective_only(x_c)
            if val_c < best_val:
                best_val = val_c
                best_x = x_c.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final polish with very high beta
    for beta in [300.0, 600.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-14})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Coordinate descent refinement with adaptive step size
    step_size = 0.005
    no_improve = 0
    max_no_improve = 5000
    patience = 0
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = rng.integers(2, 30)
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0005, step_size * 0.95)
            no_improve = 0
            patience = 0
        else:
            no_improve += 1
            patience += 1
            if patience > 500:
                step_size = min(0.02, step_size * 1.02)
                patience = 0
            else:
                step_size = min(0.02, step_size * 1.01)
    
    return best_x
# EVOLVE-BLOCK-END
