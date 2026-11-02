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
    L1 = sum_{k=-(N-1)}^{N-1} |R(k)|
    """
    corr = np.correlate(signs, signs, mode='full')
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
    
    # Pre-calculate grid x in [0, 1)
    x = np.arange(N)
    
    # Try a range of frequencies. 
    # The aperiodic autocorrelation L1 norm is minimized when the signal 
    # has balanced positive and negative segments and low spectral leakage.
    # Low frequencies (k=1, 2, ...) are key.
    # We will test k from 1 to 50.
    for k in range(1, 51):
        # For each frequency, the sign pattern is determined by the phase.
        # Since the signal is periodic in the index (conceptually), 
        # shifting the phase by 2*pi/k * m is equivalent to shifting the index by m.
        # However, for a finite sequence of length N, the aperiodic autocorrelation
        # is sensitive to the boundary effects.
        # 
        # The sign pattern is sign(sin(2*pi*k*i/N + phi)).
        # The pattern repeats every N/k samples (if N/k is integer).
        # For a given k, there are effectively N distinct phase shifts, but many
        # are equivalent due to the periodicity of the sine wave and the symmetry
        # of the autocorrelation.
        #
        # A more efficient approach:
        # The sign pattern is determined by the positions of the zero crossings.
        # For a sine wave of frequency k on [0, N), there are 2k zero crossings.
        # The phase shift moves these zero crossings.
        #
        # Let's try a set of representative phases.
        # Since the autocorrelation depends on the relative positions of the
        # positive and negative blocks, and the blocks have lengths determined
        # by k, we can try phases that align the block boundaries with the
        # sequence boundaries in different ways.
        #
        # Actually, for a fixed k, the sequence is periodic with period P = N/k 
        # (if N is divisible by k). If not, it's not perfectly periodic.
        # Let's just try a good number of phases.
        # The number of distinct sign patterns for a given k is at most N.
        # But we can be smarter. The pattern is determined by the starting phase.
        # Let's try 20 phases evenly spaced.
        
        num_phases = 20
        for phase_idx in range(num_phases):
            phi = 2 * np.pi * phase_idx / num_phases
            sine_vals = np.sin(2 * np.pi * k * x / N + phi)
            signs = np.where(sine_vals >= 0, 1.0, -1.0)
            
            score = _aperiodic_autocorr_l1(signs)
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
    
    # Also consider the constant function
    const_signs = np.ones(N)
    const_score = _aperiodic_autocorr_l1(const_signs)
    if const_score < best_score:
        best_signs = const_signs
        best_score = const_score

    return best_signs
