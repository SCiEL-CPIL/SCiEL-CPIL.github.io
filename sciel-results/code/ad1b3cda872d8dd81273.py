"""
Third autocorrelation inequality: choose a signed step function with 400 equal intervals on [-1/4, 1/4]
to improve an autoconvolution upper bound.

The task is to minimize the upper bound C3, which is related to the autoconvolution of the step function.
We model the step function as a binary sign vector derived from the zero crossings of a low-frequency sine wave.
This spectral approach shapes the phase and amplitude to exploit the symmetry of the autocorrelation kernel,
minimizing high-frequency energy in the autoconvolution.
"""

import numpy as np

N = 400

def _aperiodic_autocorr_l1(signs):
    """
    Calculate the L1 norm of the aperiodic autocorrelation.
    R(k) = sum_{i=0}^{N-1-k} signs[i] * signs[i+k]
    L1 = sum_{k=-(N-1)}^{N-1} |R(k)| = 2 * sum_{k=1}^{N-1} |R(k)| + |R(0)|
    """
    N = len(signs)
    # R(0) is sum of squares, which is N for binary signs
    r0 = N
    l1_sum = r0
    for k in range(1, N):
        # Use np.correlate or manual sum for speed?
        # np.correlate is O(N^2) if not using FFT, but for N=400 it's fine.
        # However, we are doing this in a loop.
        # Let's compute all lags at once using np.correlate
        pass
    
    # Efficient calculation using np.correlate
    # np.correlate(a, v, mode='full')
    # a is signs, v is signs
    # The result length is 2*N - 1
    # The center is R(0).
    # We need sum of abs of all lags.
    corr = np.correlate(signs, signs, mode='full')
    # corr is symmetric.
    # l1 = sum(abs(corr))
    return np.sum(np.abs(corr))

def construct():
    """
    Construct a signed step function with N equal intervals.
    Returns a sequence of signed step heights (values in {-1, +1}).
    
    Strategy: Use a low-frequency sine wave to determine the sign pattern.
    We optimize over frequency and phase to minimize the L1 norm of the
    aperiodic autoconvolution.
    """
    best_signs = np.ones(N)
    best_score = float('inf')
    
    # Pre-calculate grid x in [0, 1) or [-1/4, 1/4]?
    # The relative positions matter for aperiodic correlation.
    # Let's use indices 0 to N-1.
    x = np.arange(N)
    
    # Try low frequencies
    # Higher frequencies create more transitions, which generally increases
    # the high-frequency content of the autocorrelation, but for binary signals,
    # the aperiodic autocorrelation L1 norm behaves differently.
    # Let's try k from 1 to 100.
    for k in range(1, 101):
        # Try several phase shifts
        # The pattern is sign(sin(2*pi*k*i/N + phi))
        # Shifting i by m is equivalent to changing phi.
        # We can just try m = 0, 1, ..., M-1 shifts.
        # Or just try a few phases.
        # Since the signal is binary, there are at most N distinct shifts.
        # But many shifts yield the same L1 norm if the signal has symmetries.
        # Let's try 10 evenly spaced phases.
        for phase_idx in range(10):
            phi = 2 * np.pi * phase_idx / 10.0
            sine_vals = np.sin(2 * np.pi * k * x / N + phi)
            signs = np.where(sine_vals >= 0, 1.0, -1.0)
            
            score = _aperiodic_autocorr_l1(signs)
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
    
    # Also consider the constant function (k=0 effectively, or just ones)
    # Constant function aperiodic autocorrelation:
    # R(k) = N-k. L1 = sum_{k=0}^{N-1} (N-k) * 2 - N = N(N+1) - N = N^2.
    # For N=400, score = 160000.
    # Let's see if we can beat this.
    const_signs = np.ones(N)
    const_score = _aperiodic_autocorr_l1(const_signs)
    if const_score < best_score:
        best_signs = const_signs
        best_score = const_score

    return best_signs
