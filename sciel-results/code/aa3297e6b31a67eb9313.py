"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _compute_score(h):
    """Compute the max overlap score for a given h."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H)**2).real
    C_vals = N/2 - autocorr
    return np.max(C_vals)


def _optimize_phases(amps, phases, n_iter=200, step_size=0.05):
    """Optimize phases to minimize max |x| where x = IFFT(amps * exp(i*phases)).real"""
    phases = phases.copy()
    for it in range(n_iter):
        X = amps * np.exp(1j * phases)
        x = np.fft.ifft(X).real
        
        n_peak = np.argmax(np.abs(x))
        x_peak = x[n_peak]
        
        if abs(x_peak) < 1e-12:
            break
        
        angles_peak = phases + 2 * np.pi * np.arange(N) * n_peak / N
        grad = -amps * np.sin(angles_peak)
        
        # Adaptive step size
        if x_peak > 0:
            phases -= step_size * grad
        else:
            phases += step_size * grad
        
        phases = phases % (2 * np.pi)
    
    return phases


def _build_h_from_spectrum(amps, phases):
    """Build h from spectral amplitudes and phases, ensuring constraints."""
    X = amps * np.exp(1j * phases)
    x = np.fft.ifft(X).real
    
    # Scale to fit [-0.5, 0.5]
    max_abs = np.max(np.abs(x))
    if max_abs > 1e-10:
        scale = 0.5 / max_abs
        x = x * scale
    
    # Ensure zero mean
    x = x - np.mean(x)
    
    # Clip and re-correct
    x = np.clip(x, -0.5, 0.5)
    x = x - np.mean(x)
    x = np.clip(x, -0.5, 0.5)
    
    h = 0.5 + x
    h = np.clip(h, 0, 1)
    
    # Final sum correction
    sum_h = np.sum(h)
    if abs(sum_h - N/2) > 1e-10:
        # Distribute correction evenly, respecting bounds
        diff = N/2 - sum_h
        # Simple approach: add diff/N to each, then clip
        h = h + diff / N
        h = np.clip(h, 0, 1)
        # If clipping changed the sum, do a second pass
        sum_h2 = np.sum(h)
        if abs(sum_h2 - N/2) > 1e-10:
            diff2 = N/2 - sum_h2
            # Only adjust elements that have room
            can_inc = h < 1.0
            can_dec = h > 0.0
            if diff2 > 0 and np.any(can_inc):
                h[can_inc] += diff2 / np.sum(can_inc)
            elif diff2 < 0 and np.any(can_dec):
                h[can_dec] += diff2 / np.sum(can_dec)
            h = np.clip(h, 0, 1)
    
    return h


def construct():
    """
    Spectral design: represent h as 0.5 + x where x is zero-mean.
    Use various power spectra and optimize phases to minimize time-domain peak,
    allowing maximum scaling while maintaining the [0,1] constraint.
    """
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Try different spectral shapes
    # 1. Gaussian spectra with varying widths
    # 2. Raised cosine spectra
    # 3. Flat-top with roll-off
    
    j = np.arange(N)
    
    # Gaussian spectra
    for sigma in np.linspace(0.8, 6.0, 25):
        amps = np.exp(-(j**2) / (2 * sigma**2))
        amps += np.exp(-((j - N)**2) / (2 * sigma**2))
        amps += np.exp(-((j + N)**2) / (2 * sigma**2))
        amps[0] = 0
        
        # Normalize amps
        amps = amps / np.max(amps)
        
        # Try multiple random initializations
        for trial in range(8):
            phases = np.random.uniform(0, 2 * np.pi, N)
            phases[0] = 0  # Fix DC phase
            
            # Optimize phases
            phases_opt = _optimize_phases(amps, phases, n_iter=150, step_size=0.08)
            
            # Build h and compute score
            h = _build_h_from_spectrum(amps, phases_opt)
            score = _compute_score(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Raised cosine spectra
    for beta in np.linspace(0.1, 1.0, 10):
        # Raised cosine in frequency domain
        for k in range(N):
            # Map k to frequency in [-N/2, N/2]
            f = k if k <= N/2 else k - N
            if abs(f) <= beta * N / 2:
                amps[k] = 1.0
            elif abs(f) <= (1 + beta) * N / 2:
                amps[k] = 0.5 * (1 + np.cos(np.pi * (abs(f) - beta * N / 2) / (beta * N / 2)))
            else:
                amps[k] = 0.0
        amps[0] = 0
        
        if np.max(amps) > 0:
            amps = amps / np.max(amps)
            
            for trial in range(5):
                phases = np.random.uniform(0, 2 * np.pi, N)
                phases[0] = 0
                
                phases_opt = _optimize_phases(amps, phases, n_iter=150, step_size=0.08)
                
                h = _build_h_from_spectrum(amps, phases_opt)
                score = _compute_score(h)
                
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Final correction to ensure exact sum
    sum_h = np.sum(best_h)
    if abs(sum_h - N/2) > 1e-10:
        best_h = best_h - (sum_h - N/2) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
