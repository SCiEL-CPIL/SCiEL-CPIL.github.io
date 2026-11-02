"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_autocorr_max_abs(x):
    """
    Compute the maximum absolute value of the linear autocorrelation of x.
    Uses FFT for O(N log N) complexity.
    """
    # To compute linear autocorrelation via FFT, we zero-pad to avoid circular convolution artifacts.
    # The length of linear autocorrelation is 2*N - 1.
    # We need a transform size M >= 2*N - 1.
    # A power of 2 is efficient for FFT.
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    # Compute FFT of x and its conjugate (for real x, conjugate is x itself, but we need to handle signs)
    # Autocorrelation r[k] = sum_{i} x[i] * x[i+k]
    # This is the inverse FFT of |FFT(x)|^2.
    # Wait, the standard property is:
    # IFFT(FFT(x) * conj(FFT(x))) gives circular autocorrelation.
    # For linear autocorrelation, we zero-pad.
    
    X = np.fft.fft(x, M)
    # Power spectrum
    P = X * np.conj(X)
    # Inverse FFT
    ac = np.fft.ifft(P).real
    
    # The first N values of ac correspond to lags 0 to N-1.
    # Due to symmetry, ac[k] for k >= 0 is the correlation.
    # We want max absolute value.
    return np.max(np.abs(ac[:N]))

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], N).astype(np.float64)
    
    # Simulated Annealing
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    current_val = _fast_autocorr_max_abs(x)
    best_val = current_val
    best_x = x.copy()
    
    while time.time() < deadline:
        # Select a random index to flip
        i = random.randint(0, N - 1)
        
        # Flip the bit
        x[i] *= -1
        
        # Evaluate new state
        new_val = _fast_autocorr_max_abs(x)
        
        # Acceptance criterion
        if new_val < current_val:
            # Always accept improvements
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            # Accept worse solutions with probability exp(-delta / T)
            delta = new_val - current_val
            if random.random() < np.exp(-delta / T):
                current_val = new_val
            else:
                # Reject, revert
                x[i] *= -1
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If temperature is very low, we might be stuck in local minima, 
        # but SA is designed to escape them probabilistically.
        # To avoid infinite loops at very low T, we could restart or just let it run.
        # Given the time limit, letting it run is fine.
        
    return best_x.astype(np.int8)
# EVOLVE-BLOCK-END
