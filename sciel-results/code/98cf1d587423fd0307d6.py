"""
Third autocorrelation inequality: choose a signed step function with 400 equal intervals on [-1/4, 1/4]
to improve an autoconvolution upper bound.

The task is to minimize the upper bound C3, which is related to the autoconvolution of the step function.
We model the step function as a binary sign vector derived from the zero crossings of a low-frequency sine wave.
This spectral approach shapes the phase and amplitude to exploit the symmetry of the autocorrelation kernel,
minimizing high-frequency energy in the autoconvolution.
"""

import numpy as np
from scipy.fft import fft, ifft

N = 400


def construct():
    """
    Construct a signed step function with N equal intervals.
    Returns a sequence of signed step heights (values in {-1, +1}).
    
    Strategy: Use a low-frequency sine wave to determine the sign pattern.
    The zero crossings of the sine wave define where the sign flips.
    We choose the frequency to minimize the L1 norm of the autoconvolution
    (or a related upper bound C3).
    
    The autoconvolution of a step function f is (f * f)(t) = integral f(s) f(t-s) ds.
    For a discrete step function with N intervals, we can compute the discrete autoconvolution.
    
    We want to minimize an upper bound on C3, which is typically related to the maximum
    or L1 norm of the autoconvolution. A common bound involves the sum of absolute values
    of the autoconvolution coefficients.
    
    Using a sine wave with frequency k (number of periods over the interval), the sign pattern
    is sign(sin(2*pi*k*x/N)) where x = 0, 1, ..., N-1.
    
    We try several low frequencies and pick the one that gives the best (lowest) score.
    """
    # Define the grid
    x = np.arange(N)
    
    best_signs = None
    best_score = float('inf')
    
    # Try low frequencies k = 1, 2, 3, ... up to some limit
    for k in range(1, 20):
        # Generate the sine wave
        sine_vals = np.sin(2 * np.pi * k * x / N)
        
        # Determine signs: +1 where sine >= 0, -1 where sine < 0
        # At zero crossings, we need a consistent rule; use >= 0 for +1
        signs = np.where(sine_vals >= 0, 1.0, -1.0)
        
        # Compute the autoconvolution
        # For discrete step functions, the autoconvolution can be computed via FFT
        # The autoconvolution of f with itself is IFFT(|FFT(f)|^2)
        F = fft(signs)
        autoconv = ifft(np.abs(F)**2).real
        
        # The upper bound C3 is often related to the L1 norm or max of the autoconvolution
        # Let's use the L1 norm of the autoconvolution as our score (lower is better)
        score = np.sum(np.abs(autoconv))
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
    
    # Also try the constant function as a baseline
    const_signs = np.ones(N)
    F = fft(const_signs)
    autoconv = ifft(np.abs(F)**2).real
    const_score = np.sum(np.abs(autoconv))
    
    if const_score < best_score:
        best_signs = const_signs
    
    return best_signs
