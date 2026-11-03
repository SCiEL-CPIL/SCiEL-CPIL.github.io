import numpy as np
import time

def _autoconv(f):
    """Compute autoconvolution c_k = sum_i f_i * f_{i+k} for k in [-n+1, n-1].
    Returns array of length 2n-1 where index n-1 corresponds to k=0."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    ac_full = np.fft.ifft(F_full * np.conj(F_full))[:n].real
    full_ac = np.zeros(2*n-1)
    full_ac[n-1:] = ac_full
    full_ac[:n-1] = ac_full[1:][::-1]
    return full_ac

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    c0 = np.dot(f, f)
    if c0 == 0:
        return 0.0
    ac = _autoconv(f)
    max_ac = np.max(ac)
    return max_ac / c0

def _spectral_coordinate_descent(f_init, max_iter=500, time_limit=100.0):
    """Coordinate descent in frequency domain with soft-plus projection."""
    n = len(f_init)
    start_time = time.time()
    
    # Initialize frequency coefficients
    F = np.fft.fft(f_init)
    
    # We'll work with amplitude and phase
    amp = np.abs(F)
    phase = np.angle(F)
    
    best_f = f_init.copy()
    best_score = _compute_score(f_init)
    
    def amp_to_f(amp, phase):
        """Convert amplitude/phase to time domain, enforce non-negativity via soft-plus."""
        F_recon = amp * np.exp(1j * phase)
        f_raw = np.fft.ifft(F_recon).real
        # Soft-plus projection: f_i = max(0, f_raw_i)
        f = np.maximum(f_raw, 0)
        return f
    
    def compute_from_amp(amp, phase):
        """Compute score from amplitude/phase representation."""
        f = amp_to_f(amp, phase)
        if np.all(f == 0):
            return 0.0
        return _compute_score(f)
    
    # Coordinate descent on amplitude coefficients
    lr = 0.1
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for k in range(n):
            old_amp = amp[k]
            if old_amp < 1e-10:
                continue
            
            # Try different amplitudes for this frequency
            best_local_score = compute_from_amp(amp, phase)
            best_local_amp = old_amp
            
            for factor in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
                test_amp = old_amp * factor
                amp[k] = test_amp
                score = compute_from_amp(amp, phase)
                if score < best_local_score - 1e-12:
                    best_local_score = score
                    best_local_amp = test_amp
            
            amp[k] = best_local_amp
            if abs(best_local_amp - old_amp) > 1e-10:
                improved = True
        
        # Update best
        f_current = amp_to_f(amp, phase)
        score_current = compute_from_amp(amp, phase)
        if score_current < best_score - 1e-12:
            best_score = score_current
            best_f = f_current.copy()
        
        if not improved:
            # Try phase adjustments
            phase_improved = False
            for k in range(n):
                old_phase = phase[k]
                if amp[k] < 1e-10:
                    continue
                
                best_local_score = compute_from_amp(amp, phase)
                best_local_phase = old_phase
                
                for dphase in [-0.5, -0.2, 0.2, 0.5]:
                    test_phase = old_phase + dphase
                    phase[k] = test_phase
                    score = compute_from_amp(amp, phase)
                    if score < best_local_score - 1e-12:
                        best_local_score = score
                        best_local_phase = test_phase
                
                phase[k] = best_local_phase
                if abs(best_local_phase - old_phase) > 1e-10:
                    phase_improved = True
            
            if not phase_improved:
                break
    
    return best_f, best_score

def _random_spectral_init(n, n_trials=50):
    """Generate random spectral initializations."""
    best_f = np.ones(n)
    best_score = _compute_score(best_f)
    
    for trial in range(n_trials):
        # Random phase with fixed amplitude spectrum
        amp = np.ones(n)
        phase = 2 * np.pi * np.random.rand(n)
        F = amp * np.exp(1j * phase)
        f_raw = np.fft.ifft(F).real
        f = np.maximum(f_raw, 0)
        
        if np.any(f > 0):
            score = _compute_score(f)
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    return best_f, best_score

def construct():
    n = 600
    # Use a Cosine Bell shape, which is a strong candidate for minimizing
    # the peak of the autoconvolution relative to its energy (or other norms).
    # f_i = max(0, cos(pi * i / n)) for i in [0, n-1]
    # This corresponds to a semi-circular pulse on the interval.
    i = np.arange(n)
    f = np.cos(np.pi * i / n)
    # Ensure non-negative (cos(pi) is -1, but we stop before pi? 
    # i goes 0 to 599. pi * 599/600 is close to pi. cos is negative.
    # We want a pulse on [-1/4, 1/4].
    # The intervals are equal.
    # A standard "good" function is a Gaussian or Cosine Bell centered at 0.
    # Let's center it.
    # x from -0.25 to 0.25.
    # f(x) = max(0, cos(2 * pi * x)) ? No, that's a full cosine.
    # Let's use a raised cosine (Hann window) shape, which is non-negative.
    # f(x) = 0.5 * (1 + cos(2 * pi * x / width))
    # If width = 0.5 (full interval), it's a single hump.
    # x = -0.25 + i/600 * 0.5
    x = -0.25 + (i / n) * 0.5
    # Hann window shape centered at 0, width 0.5
    f = 0.5 * (1 + np.cos(2 * np.pi * x / 0.5))
    # Clip to ensure non-negative (numerical precision)
    f = np.maximum(f, 0)
    
    # Optional: Simple local refinement if time permits (but likely not needed)
    # The prompt emphasizes fixing the timeout. A fast, good initial guess is sufficient.
    
    return list(f)
