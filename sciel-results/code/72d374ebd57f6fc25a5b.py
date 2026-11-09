"""Spectral optimization for Erdos minimum overlap."""
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 100

def _overlap_max(h):
    """Compute max overlap of h with shifted 1-h over linear shifts."""
    one_minus_h = 1.0 - h
    overlaps = np.correlate(h, one_minus_h, mode='valid')
    return np.max(overlaps)

def _spectral_overlap(h):
    """
    Compute the overlap function using FFT for speed.
    Returns the max overlap value.
    """
    n = len(h)
    # Use FFT to compute cross-correlation
    f_h = np.fft.rfft(h, n=2*n)
    f_omh = np.fft.rfft(1.0 - h, n=2*n)
    # Cross-correlation: corr[k] = sum_i h[i] * (1-h)[i+k]
    # This is the 'valid' mode of correlate
    corr = np.fft.irfft(f_h * np.conj(f_omh), n=2*n)
    # Extract the valid part
    # For 'valid' mode with mode='valid' in correlate, we need indices 0 to n-1
    # Actually, let me think about this more carefully.
    # np.correlate(h, g, 'valid') gives sum_{i=0}^{n-1-k} h[i]*g[i+k] for k=0..n-1
    # Using FFT: cross-correlation of h and g is ifft(fft(h)*conj(fft(g)))
    # But we need to be careful about the indexing.
    # Let's just use np.correlate directly for correctness, it's fast enough for N=100
    one_minus_h = 1.0 - h
    overlaps = np.correlate(h, one_minus_h, mode='valid')
    return np.max(overlaps)

def _construct_sinusoidal(freq, amplitude, phase):
    """
    Construct a step function h[i] = 0.5 + amplitude * sin(2*pi*freq*i/N + phase)
    Clip to [0,1] and adjust to ensure sum is exactly N/2.
    """
    i = np.arange(N)
    h = 0.5 + amplitude * np.sin(2.0 * np.pi * freq * i / N + phase)
    h = np.clip(h, 0.0, 1.0)
    # Adjust to ensure sum is exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    # Distribute the difference uniformly
    h = h + diff / N
    # Re-clip and re-adjust (iterative refinement)
    for _ in range(10):
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
    return h

def _try_sinusoidal_params(freq, amplitude, phase):
    """Try a specific sinusoidal parameter set."""
    h = _construct_sinusoidal(freq, amplitude, phase)
    return _overlap_max(h), h

def construct():
    """
    Returns a sequence of step heights in [0, 1] whose sum is exactly len/2.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = np.inf
    
    # Try a grid of frequencies, amplitudes, and phases
    # Frequency: try values that are "incommensurate" with N
    # For N=100, good frequencies might be around 0.3, 0.4, 0.5, etc.
    # The key is to avoid frequencies that are integer multiples of 1/N
    
    freqs = np.linspace(0.1, 0.9, 17)  # 17 frequency values
    amplitudes = np.linspace(0.1, 0.5, 9)  # 9 amplitude values
    phases = np.linspace(0, 2*np.pi, 13, endpoint=False)  # 13 phase values
    
    # First pass: coarse grid search
    for f in freqs:
        if time.time() - start_time > deadline:
            break
        for a in amplitudes:
            if time.time() - start_time > deadline:
                break
            for p in phases:
                if time.time() - start_time > deadline:
                    break
                score, h = _try_sinusoidal_params(f, a, p)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Second pass: local refinement around the best found parameters
    if best_h is not None and time.time() - start_time < deadline:
        # We need to track which parameters gave the best score
        # Let's do a finer search around promising regions
        # Try more frequencies around the best ones
        best_freq = None
        best_amp = None
        best_phase = None
        best_found_score = np.inf
        
        for f in freqs:
            if time.time() - start_time > deadline:
                break
            for a in amplitudes:
                if time.time() - start_time > deadline:
                    break
                for p in phases:
                    if time.time() - start_time > deadline:
                        break
                    score, h = _try_sinusoidal_params(f, a, p)
                    if score < best_found_score:
                        best_found_score = score
                        best_freq = f
                        best_amp = a
                        best_phase = p
        
        if best_freq is not None:
            # Fine-tune around the best parameters
            freq_fine = np.linspace(best_freq - 0.1, best_freq + 0.1, 11)
            amp_fine = np.linspace(max(0.05, best_amp - 0.1), min(0.6, best_amp + 0.1), 11)
            phase_fine = np.linspace(best_phase - 0.5, best_phase + 0.5, 11)
            
            for f in freq_fine:
                if time.time() - start_time > deadline:
                    break
                for a in amp_fine:
                    if time.time() - start_time > deadline:
                        break
                    for p in phase_fine:
                        if time.time() - start_time > deadline:
                            break
                        score, h = _try_sinusoidal_params(f, a, p)
                        if score < best_score:
                            best_score = score
                            best_h = h.copy()
    
    # If we didn't find anything good, fall back to a simple construction
    if best_h is None:
        # Use a binary sequence with alternating pattern
        h = np.zeros(N)
        h[::2] = 1.0
        # This gives sum = N/2
        best_h = h
        best_score = _overlap_max(h)
    
    # Verify the sum constraint
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-10:
        # Force adjustment
        diff = N/2.0 - current_sum
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-adjust
        for _ in range(20):
            current_sum = np.sum(best_h)
            diff = N/2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            best_h = best_h + diff / N
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
