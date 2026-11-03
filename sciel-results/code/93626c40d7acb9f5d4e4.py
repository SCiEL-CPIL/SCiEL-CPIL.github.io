"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parameterize the step function by its DFT coefficients (magnitude and phase).
2. Enforce non-negativity via soft-plus projection in the time domain.
3. Directly minimize the maximum value of the autoconvolution (which is the inverse DFT of |F|^2)
   using coordinate descent on the frequency domain.
4. This decouples the peak constraint from the time-domain shape search, allowing
   direct control over the autoconvolution profile via spectral manipulation.

Key improvements over seed:
- Spectral representation allows direct manipulation of the autoconvolution peak
- Coordinate descent on frequency bins is efficient and can target specific lags
- Soft-plus projection maintains smoothness while enforcing non-negativity
- Adaptive step sizes for faster convergence
"""

import numpy as np
from scipy.fft import fft, ifft
import time

N = 600
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Autoconvolution: (f*f)(z) = sum_j f(j) f(j-k) for discrete
    # Using FFT: autoconv = ifft(fft(f) * fft(f))
    F = fft(f)
    autoconv = ifft(F * F).real
    max_conv = np.max(autoconv)
    return max_conv / (s * s)

def softplus(x, beta=10.0):
    """Softplus activation for smooth non-negativity."""
    return np.log1p(np.exp(beta * x)) / beta

def softplus_grad(x, beta=10.0):
    """Gradient of softplus."""
    return 1.0 / (1.0 + np.exp(-beta * x))

def spectral_to_time(magnitudes, phases, beta=10.0):
    """Convert spectral parameters to time-domain step function with non-negativity."""
    # Reconstruct complex DFT
    F = magnitudes * np.exp(1j * phases)
    # Inverse DFT
    f = ifft(F).real
    # Apply softplus to enforce non-negativity
    f = softplus(f, beta)
    return f

def compute_autoconv_from_spectral(magnitudes, phases):
    """Compute autoconvolution directly from spectral representation."""
    F = magnitudes * np.exp(1j * phases)
    # Autoconvolution in frequency domain is |F|^2
    # In time domain: ifft(|F|^2)
    power_spectrum = np.abs(F) ** 2
    autoconv = ifft(power_spectrum).real
    return autoconv

def construct():
    start_time = time.time()
    
    # Initialize with a reasonable starting point
    # Start with a smooth function in frequency domain
    # Low frequencies should be dominant for smooth time-domain function
    
    # Initialize magnitudes with a smooth decay
    freqs = np.fft.fftfreq(N)
    # Smooth decay: low frequencies high, high frequencies low
    magnitudes = np.exp(-2.0 * np.abs(freqs) * N * 0.1)
    # Ensure DC component is reasonable
    magnitudes[0] = np.sum(magnitudes) * 0.5
    
    # Initialize phases to zero (symmetric function)
    phases = np.zeros(N)
    
    # Compute initial score
    f_initial = spectral_to_time(magnitudes, phases, beta=10.0)
    best_f = f_initial.copy()
    best_score = compute_score(best_f)
    
    # Coordinate descent on frequency domain
    # We'll optimize the magnitudes and phases separately
    
    alpha_mag = 0.1
    alpha_phase = 0.05
    beta_sp = 10.0
    
    n_iter = 0
    while time.time() - start_time < TIME_LIMIT - 5.0:
        n_iter += 1
        improved = False
        
        # Update magnitudes
        for i in range(N):
            if time.time() - start_time > TIME_LIMIT - 5.0:
                break
            
            old_mag = magnitudes[i]
            # Try a few candidate values
            candidates = [0.0, old_mag * 0.7, old_mag * 0.9, old_mag, old_mag * 1.1, old_mag * 1.3, 1.0]
            
            best_candidate = old_mag
            best_candidate_score = best_score
            
            for cand in candidates:
                magnitudes[i] = max(0.0, cand)
                f = spectral_to_time(magnitudes, phases, beta_sp)
                score = compute_score(f)
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = cand
            
            magnitudes[i] = best_candidate
            if best_candidate != old_mag:
                improved = True
                best_score = best_candidate_score
                best_f = spectral_to_time(magnitudes, phases, beta_sp).copy()
        
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
        
        # Update phases
        for i in range(N):
            if time.time() - start_time > TIME_LIMIT - 5.0:
                break
            
            old_phase = phases[i]
            # Try a few candidate phase values
            candidates = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
            
            best_candidate = old_phase
            best_candidate_score = best_score
            
            for cand in candidates:
                phases[i] = cand
                f = spectral_to_time(magnitudes, phases, beta_sp)
                score = compute_score(f)
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = cand
            
            phases[i] = best_candidate
            if best_candidate != old_phase:
                improved = True
                best_score = best_candidate_score
                best_f = spectral_to_time(magnitudes, phases, beta_sp).copy()
        
        if not improved:
            # Try adjusting beta for softplus
            beta_sp *= 0.9
            if beta_sp < 1.0:
                break
        
        # Adaptive step size
        if improved:
            alpha_mag = min(alpha_mag * 1.05, 0.5)
            alpha_phase = min(alpha_phase * 1.05, 0.3)
        else:
            alpha_mag *= 0.9
            alpha_phase *= 0.9
    
    # Final refinement: local coordinate descent in time domain
    f = best_f.copy()
    for i in range(N):
        if time.time() - start_time > TIME_LIMIT - 2.0:
            break
        
        old_val = f[i]
        candidates = [0.0, old_val * 0.5, old_val * 0.8, old_val, old_val * 1.2, old_val * 1.5]
        
        best_candidate = old_val
        best_candidate_score = best_score
        
        for cand in candidates:
            f[i] = max(0.0, cand)
            score = compute_score(f)
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = cand
        
        f[i] = best_candidate
        if best_candidate != old_val:
            best_score = best_candidate_score
            best_f = f.copy()
    
    # Ensure non-negativity
    best_f = np.maximum(best_f, 0.0)
    
    # Convert to list
    return [float(x) for x in best_f]
