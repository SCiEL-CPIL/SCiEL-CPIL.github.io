"""Chebyshev spectral envelope design + phase retrieval for C3' minimization."""
import numpy as np
import time

N = 400
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
    """Compute objective value only."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    m = np.max(np.abs(ac))
    return m / (s * s)

def _phase_retrieval(mag_target, rng, iters=2000):
    """
    Phase retrieval: find phase phi such that |IFFT(mag_target * exp(i*phi))| 
    approximates a time-domain signal with desired properties.
    We want the time-domain signal to have its energy concentrated in the first N samples.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    # Initialize with random phase
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0.0  # DC real
    
    x = np.zeros(N)
    
    for it in range(iters):
        if _check_deadline():
            break
        
        # Build spectrum with current phase
        spec = mag_target.copy()
        # Apply phase (for rfft, we need to handle conjugate symmetry)
        # For a real signal, X[k] = conj(X[N-k])
        # We work with the full spectrum conceptually but use rfft
        
        # Full complex spectrum
        full_spec = np.zeros(L_fft, dtype=complex)
        full_spec[:L_fft//2+1] = spec * np.exp(1j * phase)
        for k in range(1, L_fft//2):
            full_spec[L_fft - k] = np.conj(full_spec[k])
        
        # Inverse FFT
        x_full = np.fft.ifft(full_spec).real
        
        # Constraint: x should be zero outside [0, N)
        # Keep only first N samples, zero the rest
        x_full[N:] = 0
        if N > 1:
            x_full[:0] = 0  # no-op, just for clarity
        
        # Forward FFT to get new spectrum
        new_spec = np.fft.fft(x_full)
        
        # Extract magnitude and phase
        new_phase = np.angle(new_spec[:L_fft//2+1])
        new_phase[0] = 0.0
        
        # Update phase (relaxation)
        alpha = 0.9  # relaxation factor
        phase = alpha * phase + (1 - alpha) * new_phase
        phase[0] = 0.0
        
        # Compute current x
        full_spec2 = np.zeros(L_fft, dtype=complex)
        full_spec2[:L_fft//2+1] = mag_target * np.exp(1j * phase)
        for k in range(1, L_fft//2):
            full_spec2[L_fft - k] = np.conj(full_spec2[k])
        x_full2 = np.fft.ifft(full_spec2).real
        x = x_full2[:N]
        
        # Ensure positive sum
        if np.sum(x) < 0:
            x = -x
    
    # Final computation
    full_spec = np.zeros(L_fft, dtype=complex)
    full_spec[:L_fft//2+1] = mag_target * np.exp(1j * phase)
    for k in range(1, L_fft//2):
        full_spec[L_fft - k] = np.conj(full_spec[k])
    x_full = np.fft.ifft(full_spec).real
    x = x_full[:N]
    if np.sum(x) < 0:
        x = -x
    
    return x

def _chebyshev_envelope_optimize(rng, iters=500):
    """
    Remez-like exchange algorithm to find spectral envelope E(f) that minimizes
    the L_infinity norm of the periodic autoconvolution (IFFT of |E|^2).
    
    The periodic autoconvolution is: ac_per[k] = IFFT(|E|^2)[k]
    We want to minimize max_k |ac_per[k]| subject to sum constraint.
    
    Strategy: start with flat envelope, iteratively adjust to reduce peaks.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Initialize with flat envelope
    E = np.ones(n_freq)
    
    # The periodic autoconvolution magnitude squared in frequency is |E|^2
    # ac_per = IFFT(|E|^2)
    # We want to minimize max |ac_per|
    
    # Exchange algorithm: identify peak lags, adjust E to reduce them
    for it in range(iters):
        if _check_deadline():
            break
        
        # Compute periodic autoconvolution
        E_sq = E * E
        ac_per = np.fft.ifft(E_sq).real  # length L_fft
        
        # Find the peak
        abs_ac = np.abs(ac_per)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        # The gradient of ac_per[k] w.r.t. E[m] is:
        # ac_per[k] = (1/L) sum_m E[m]^2 * exp(2*pi*i*m*k/L)
        # d(ac_per[k])/d(E[m]) = (2/L) E[m] * exp(2*pi*i*m*k/L)
        
        # To reduce the peak at lag k, we should decrease E[m] where 
        # E[m] * exp(2*pi*i*m*k/L) has the same sign as ac_per[k]
        
        # Simple gradient step
        k = peak_idx
        grad_E = 2.0 / L_fft * E * np.exp(2j * np.pi * k * np.arange(n_freq) / L_fft)
        
        # Only update the positive-frequency part (E is real and non-negative)
        # The real part of the gradient tells us how to adjust
        grad_real = np.real(grad_E)
        
        # Step: decrease E where gradient is positive (increases the peak)
        step = 0.01
        E = E - step * grad_real
        E = np.maximum(0.01, E)  # Keep positive
        
        # Normalize to maintain sum constraint approximately
        # The integral (sum) of x is related to E[0]
        # Actually, let's normalize E so that the DC component gives sum ~ 1
        # sum(x) = x[0] + ... but in frequency, sum(x) = X[0] = E[0] * exp(i*phase[0])
        # So E[0] controls the sum. Let's keep E[0] fixed.
    
    return E

def _spectral_flat_top(mag, center_frac=0.5, width_frac=0.8):
    """Create a flat-top magnitude spectrum."""
    n = len(mag)
    center = int(n * center_frac)
    half_width = int(n * width_frac / 2)
    mag_new = np.zeros(n)
    lo = max(0, center - half_width)
    hi = min(n, center + half_width + 1)
    mag_new[lo:hi] = 1.0
    return mag_new

def _spectral_gaussian(mag, center_frac=0.5, sigma_frac=0.2):
    """Create a Gaussian-tapered magnitude spectrum."""
    n = len(mag)
    freqs = np.arange(n) / n
    center = center_frac
    sigma = sigma_frac
    return np.exp(-0.5 * ((freqs - center) / sigma) ** 2)

def _spectral_raised_cosine(mag, center_frac=0.5, width_frac=0.6):
    """Create a raised-cosine magnitude spectrum."""
    n = len(mag)
    freqs = np.arange(n) / n
    center = center_frac
    half_width = width_frac / 2
    mag_new = np.zeros(n)
    for i, f in enumerate(freqs):
        d = abs(f - center)
        if d < half_width:
            mag_new[i] = 0.5 * (1 + np.cos(np.pi * d / half_width))
    return mag_new

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    
    # Generate various spectral envelope designs
    envelope_designs = []
    
    # 1. Flat-top with various widths and centers
    for width in [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for center in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mag = np.zeros(n_freq)
            mag_des = _spectral_flat_top(mag, center, width)
            envelope_designs.append(mag_des.copy())
    
    # 2. Gaussian with various sigmas
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        for center in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mag = np.zeros(n_freq)
            mag_des = _spectral_gaussian(mag, center, sigma)
            envelope_designs.append(mag_des.copy())
    
    # 3. Raised cosine with various parameters
    for width in [0.4, 0.6, 0.8, 1.0]:
        for center in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mag = np.zeros(n_freq)
            mag_des = _spectral_raised_cosine(mag, center, width)
            envelope_designs.append(mag_des.copy())
    
    # 4. Uniform (all ones)
    envelope_designs.append(np.ones(n_freq))
    
    # 5. Low-pass
    for cutoff in [0.2, 0.3, 0.4, 0.5]:
        mag = np.zeros(n_freq)
        mag[:int(n_freq * cutoff)] = 1.0
        envelope_designs.append(mag.copy())
    
    # 6. High-pass
    for cutoff in [0.3, 0.4, 0.5]:
        mag = np.zeros(n_freq)
        mag[int(n_freq * cutoff):] = 1.0
        envelope_designs.append(mag.copy())
    
    # 7. Band-pass
    for lo, hi in [(0.2, 0.5), (0.3, 0.6), (0.2, 0.4), (0.4, 0.7)]:
        mag = np.zeros(n_freq)
        mag[int(n_freq * lo):int(n_freq * hi)] = 1.0
        envelope_designs.append(mag.copy())
    
    # Try phase retrieval for each envelope design
    for mag_des in envelope_designs:
        if _check_deadline():
            break
        
        # Ensure non-negative
        mag_des = np.maximum(0, mag_des)
        
        # Normalize so that DC component gives reasonable sum
        if mag_des[0] > 0:
            mag_des = mag_des / mag_des[0] * 1.0
        
        # Phase retrieval
        x = _phase_retrieval(mag_des, rng, iters=1500)
        
        # Ensure positive sum
        if np.sum(x) < 0:
            x = -x
        
        val = _objective_only(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    if _check_deadline():
        return best_x
    
    # Now do Chebyshev-like refinement on the best envelope
    # Get the spectrum of the best x
    X_best = np.fft.rfft(best_x, n=L_fft)
    mag_best = np.abs(X_best)
    mag_best = np.maximum(0.01, mag_best)
    
    # Remez-like refinement: iteratively adjust the envelope
    E = mag_best.copy()
    
    for it in range(300):
        if _check_deadline():
            break
        
        # Compute periodic autoconvolution
        E_sq = E * E
        ac_per = np.fft.ifft(E_sq).real
        
        abs_ac = np.abs(ac_per)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        if peak_val < 1e-10:
            break
        
        # Gradient step to reduce the peak
        k = peak_idx
        grad_E = (2.0 / L_fft) * E * np.exp(2j * np.pi * k * np.arange(n_freq) / L_fft)
        grad_real = np.real(grad_E)
        
        # Adaptive step size
        step = 0.005 * (peak_val / (np.mean(abs_ac) + 1e-10))
        step = min(step, 0.05)
        
        E = E - step * grad_real
        E = np.maximum(0.01, E)
        
        # Phase retrieval with updated envelope
        x = _phase_retrieval(E, rng, iters=800)
        
        if np.sum(x) < 0:
            x = -x
        
        val = _objective_only(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    if _check_deadline():
        return best_x
    
    # L-BFGS polish on the best solution
    from scipy.optimize import minimize
    
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
        
        # Compute grad_x via FFT correlation
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
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        soft_val, grad_soft = _soft_max_and_grad(x, beta)
        val = soft_val / (s * s)
        grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
        return val, grad
    
    for beta in [50.0, 100.0, 200.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Coordinate descent refinement
    step_size = 0.003
    no_improve = 0
    max_no_improve = 3000
    
    while no_improve < max_no_improve and not _check_deadline():
        n_perturb = int(rng.integers(2, 20))
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = np.zeros(N)
        perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0003, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.015, step_size * 1.01)
    
    return best_x
