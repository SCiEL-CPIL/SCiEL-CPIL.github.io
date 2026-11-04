"""
Strategy: Zadoff-Chu based chirp design with 1D scalar optimization.

Rationale:
1. Zadoff-Chu sequences have ideal autocorrelation properties (zero off-peak autocorrelation).
2. The real part of a complex Zadoff-Chu sequence with optimized quadratic phase 
   provides a low-dimensional parameterization (just the chirp rate).
3. This bypasses high-dimensional local minima by leveraging the inherent structure 
   of constant-amplitude zero-autocorrelation (CAZAC) sequences.
4. The 1D search space is easily explored exhaustively or with fine-grained scanning.
"""

import numpy as np
import time

N = 400

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _zadoff_chu_signal(chirp_rate):
    """
    Generate a real-valued signal from the real part of a complex Zadoff-Chu-like sequence.
    
    The complex sequence is: z[n] = exp(1j * pi * chirp_rate * n * (n - 1) / N)
    We take the real part: x[n] = cos(pi * chirp_rate * n * (n - 1) / N)
    
    This is a chirp signal with quadratic phase. The chirp_rate parameter controls 
    the rate of frequency sweep.
    """
    t = np.arange(N)
    phase = np.pi * chirp_rate * t * (t - 1) / N
    x = np.cos(phase)
    
    # Ensure non-zero integral by adding a small DC offset if needed
    # The integral of a pure chirp over an integer number of periods may be near zero
    # We add a small constant to ensure the integral is non-zero
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1  # Add small DC offset
    
    return x

def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0
    
    # Initial best: try a few chirp rates
    best_x = None
    best_val = float('inf')
    
    # Exhaustive scan over chirp_rate in a reasonable range
    # The chirp rate determines the frequency sweep. For N=400, 
    # rates from 1 to 400 cover the full range of possible chirps.
    # We'll scan with fine granularity.
    
    # First, do a coarse scan
    chirp_rates_coarse = np.arange(1, 401, 5)
    for cr in chirp_rates_coarse:
        if time.time() > deadline - 10:
            break
        x = _zadoff_chu_signal(cr)
        val = _metric(x)
        if val < best_val:
            best_val = val
            best_x = x
    
    # If we have time, do a fine scan around the best coarse result
    if time.time() < deadline - 20 and best_x is not None:
        # Find the best coarse chirp rate
        best_cr_coarse = None
        best_val_coarse = float('inf')
        for cr in chirp_rates_coarse:
            x = _zadoff_chu_signal(cr)
            val = _metric(x)
            if val < best_val_coarse:
                best_val_coarse = val
                best_cr_coarse = cr
        
        if best_cr_coarse is not None:
            # Fine scan around the best coarse result
            chirp_rates_fine = np.arange(best_cr_coarse - 5, best_cr_coarse + 5, 0.1)
            for cr in chirp_rates_fine:
                if time.time() > deadline - 10:
                    break
                x = _zadoff_chu_signal(cr)
                val = _metric(x)
                if val < best_val:
                    best_val = val
                    best_x = x
    
    # If we have even more time, try a multi-component chirp: 
    # sum of two chirps with different rates
    if time.time() < deadline - 30:
        # Try combinations of two chirp rates
        best_cr1 = None
        best_cr2 = None
        best_val_multi = float('inf')
        
        # Coarse scan for first chirp
        for cr1 in np.arange(1, 100, 5):
            if time.time() > deadline - 15:
                break
            for cr2 in np.arange(100, 400, 5):
                if time.time() > deadline - 10:
                    break
                t = np.arange(N)
                x = np.cos(np.pi * cr1 * t * (t - 1) / N) + np.cos(np.pi * cr2 * t * (t - 1) / N)
                val = _metric(x)
                if val < best_val_multi:
                    best_val_multi = val
                    best_cr1 = cr1
                    best_cr2 = cr2
        
        if best_val_multi < best_val:
            t = np.arange(N)
            best_x = np.cos(np.pi * best_cr1 * t * (t - 1) / N) + np.cos(np.pi * best_cr2 * t * (t - 1) / N)
            best_val = best_val_multi
    
    # Final fallback: if no good result, return a simple chirp
    if best_x is None:
        best_x = _zadoff_chu_signal(10)
    
    return best_x
