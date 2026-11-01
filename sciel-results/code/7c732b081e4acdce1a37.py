"""
Third autocorrelation inequality: minimize the upper bound on C3
by choosing a signed step function with 400 equal intervals on [-1/4, 1/4].

The score is the raw value of the upper bound (lower is better).
The seed program returns np.ones(400), which corresponds to a constant +1 step function.

Strategy:
- The "upper bound on C3" in this context is interpreted as the maximum sidelobe
  magnitude of the autocorrelation of the step function (normalized appropriately),
  which is a standard quantity in signal processing for minimizing peak sidelobe level.
- We use a local search / simulated annealing approach over the signs of the 400 intervals.
- The autocorrelation is computed via FFT for speed.
- We track elapsed time and return the best result found before the deadline.
"""

import numpy as np
import time

N = 400

def _autocorr_sidelobe_max(signs):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes
    for a signed step function represented by a 1D array of +1/-1 values.
    
    The autocorrelation is computed as the inverse FFT of |FFT(signs)|^2.
    The main lobe is at lag 0; we take the max over all other lags.
    """
    # Zero-pad for linear autocorrelation
    n_fft = 2 * N
    f = np.fft.fft(signs, n=n_fft)
    acf = np.fft.ifft(np.abs(f) ** 2).real
    # The autocorrelation is circular; for linear, we take the first N values
    # Actually, for a real signal, the autocorrelation is symmetric.
    # The peak at lag 0 is acf[0]. Sidelobes are all other lags.
    # For linear autocorrelation of length N, we consider lags -N+1 to N-1.
    # With FFT of size 2N, the autocorrelation is:
    #   acf_full = np.fft.ifft(np.abs(f)**2).real
    # The linear autocorrelation lags are: acf_full[0:N] for lags 0 to N-1
    # and acf_full[N+1:] for lags -(N-1) to -1.
    # Let's just take the max over all indices except 0.
    acf[0] = 0  # Remove main lobe
    return np.max(np.abs(acf))


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 110s out of 120s limit
    
    # Initialize with random signs
    best_signs = np.ones(N)
    best_val = _autocorr_sidelobe_max(best_signs)
    
    # Simulated annealing
    current_signs = np.ones(N).copy()
    current_val = _autocorr_sidelobe_max(current_signs)
    
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    
    # Precompute indices for shuffling
    idx = np.arange(N)
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Try flipping
        old_val = current_signs[i]
        current_signs[i] = -old_val
        new_val = _autocorr_sidelobe_max(current_signs)
        
        delta = new_val - current_val
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_signs = current_signs.copy()
        else:
            current_signs[i] = old_val
        
        T *= cooling_rate
        
        # If temperature is too low, restart with a higher temperature
        if T < 0.01:
            T = 0.5
    
    return best_signs
