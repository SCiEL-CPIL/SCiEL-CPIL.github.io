import numpy as np
import time

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT.
    Returns array of length 2n-1."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Spectral approach: parameterize by DFT amplitudes
    # f = inverse_dft(amp * exp(i*phase))
    # Enforce non-negativity via soft-plus: f_i = max(0, raw_f_i)
    # Minimize max(f*f) / (sum f)^2
    
    # Initialize with a broad Gaussian-like shape in frequency domain
    # Low frequencies dominant -> smooth, flat-ish time domain
    k = np.arange(n)
    # Initialize amplitudes: strong DC, decaying high freq
    amp = np.exp(-k**2 / (n/4)**2)
    amp[0] = 1.0
    phase = np.zeros(n)
    
    def amp_to_f(amp, phase):
        """Convert spectral params to time-domain f (non-negative)."""
        F = amp * np.exp(1j * phase)
        f_raw = np.fft.ifft(F).real
        # Soft-plus projection for non-negativity
        f = np.maximum(f_raw, 0.0)
        return f
    
    def compute_from_amp(amp, phase):
        f = amp_to_f(amp, phase)
        score = _compute_score(f)
        return f, score
    
    f, best_score = compute_from_amp(amp, phase)
    best_amp = amp.copy()
    best_phase = phase.copy()
    
    # Coordinate descent on amplitude and phase
    # For each frequency bin, try small perturbations
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        improved = False
        
        # Amplitude coordinate descent
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            
            orig_amp = best_amp[i]
            best_local = best_score
            best_local_amp = orig_amp
            
            # Try a few perturbations
            for delta in [0.1, -0.1, 0.5, -0.5, 2.0, 0.5]:
                test_amp = best_amp.copy()
                test_amp[i] = orig_amp + delta
                f_test, score_test = compute_from_amp(test_amp, best_phase)
                if score_test < best_local:
                    best_local = score_test
                    best_local_amp = orig_amp + delta
            
            if best_local < best_score - 1e-10:
                best_amp[i] = best_local_amp
                best_score = best_local
                improved = True
        
        # Phase coordinate descent
        for i in range(1, n):  # Skip DC (phase 0)
            if time.time() > deadline - 5.0:
                break
            
            orig_phase = best_phase[i]
            best_local = best_score
            best_local_phase = orig_phase
            
            for delta in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
                test_phase = best_phase.copy()
                test_phase[i] = orig_phase + delta
                f_test, score_test = compute_from_amp(best_amp, test_phase)
                if score_test < best_local:
                    best_local = score_test
                    best_local_phase = orig_phase + delta
            
            if best_local < best_score - 1e-10:
                best_phase[i] = best_local_phase
                best_score = best_local
                improved = True
        
        if not improved:
            # Try random restart with perturbation
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                test_amp = best_amp + rng.normal(0, 0.1, n)
                test_amp = np.maximum(test_amp, 0)
                test_phase = best_phase + rng.normal(0, 0.1, n)
                f_test, score_test = compute_from_amp(test_amp, test_phase)
                if score_test < best_score:
                    best_amp = test_amp
                    best_phase = test_phase
                    best_score = score_test
                    improved = True
            if not improved:
                break
    
    f = amp_to_f(best_amp, best_phase)
    return list(f)
