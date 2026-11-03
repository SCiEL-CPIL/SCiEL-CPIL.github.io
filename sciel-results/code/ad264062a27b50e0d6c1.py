"""
Optimization of a signed step function to minimize an autocorrelation-based metric.
The task specifies minimizing an "upper bound on C3", which is interpreted here as
minimizing the L1 norm of the aperiodic autocorrelation (a standard proxy for
autocorrelation bounds in signal processing and combinatorics).

The Thue-Morse sequence is used as an initialization as it is known to have
excellent autocorrelation properties (bounded sidelobes).
A local search (greedy hill-climbing) is then applied to further minimize the metric.
"""

import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n."""
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 1
    length = 1
    while length < n:
        tm[length:2*length] = -tm[:length]
        length *= 2
    return tm[:n]

def _calc_metric(f):
    """
    Calculate the L1 norm of the aperiodic autocorrelation.
    f is a 1D array of signed values.
    """
    # Efficient autocorrelation using FFT
    # Zero-pad to next power of 2 for FFT efficiency
    n = len(f)
    size = 1 << (2 * n - 1).bit_length()
    
    # FFT of f
    F = np.fft.fft(f, n=size)
    
    # Power spectrum
    P = np.abs(F) ** 2
    
    # Inverse FFT to get autocorrelation
    # The result is circular autocorrelation. We take the first n values for aperiodic.
    # Note: np.fft.ifft returns complex, take real part.
    ac = np.fft.ifft(P).real
    
    # The aperiodic autocorrelation R_k = sum_{i=0}^{N-1-k} f_i f_{i+k}
    # The circular autocorrelation C_k = sum_{i=0}^{N-1} f_i f_{(i+k)%N}
    # For k < N, if we zero-padded correctly, the first N values of the ifft of |FFT|^2
    # correspond to the aperiodic autocorrelation?
    # Actually, ifft(|FFT(f, size)|^2) gives the circular autocorrelation of the zero-padded signal.
    # Since the signal is zero-padded to size >= 2N-1, the circular autocorrelation
    # of the zero-padded signal matches the aperiodic autocorrelation for lags 0 to N-1
    # and N to size-1-N are the negative lags.
    
    # We want sum_{k=0}^{N-1} |R_k|
    # R_k is ac[k]
    # Note: ac[0] is the energy.
    return np.sum(np.abs(ac[:n]))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Initialization: Thue-Morse
    f = _thue_morse(N).astype(np.float64)
    current_score = _calc_metric(f)
    best_f = f.copy()
    best_score = current_score
    
    # 2. Local Search: Greedy Hill Climbing
    # We perform multiple passes. In each pass, we try to flip each bit.
    # To avoid getting stuck in a very shallow local minimum, we can use a simple
    # Simulated Annealing strategy or just restart if stuck.
    # Given N=400, one pass is 400 evaluations.
    # Each evaluation is O(N log N).
    # 400 * 400 * log(800) ~ 400 * 400 * 10 = 1.6M ops per pass.
    # This is extremely fast. We can do many passes.
    
    no_improvement_count = 0
    max_no_improvement = 5
    
    # Randomize the order of indices to break symmetries and avoid cyclic patterns
    indices = np.arange(N)
    
    while time.time() < deadline:
        # Shuffle indices for this pass
        np.random.shuffle(indices)
        
        improved = False
        for i in indices:
            if time.time() > deadline:
                break
            
            # Flip bit i
            f[i] *= -1
            
            # Calculate new score
            new_score = _calc_metric(f)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                # Revert
                f[i] *= -1
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                # Try a random restart with perturbation
                # Perturb best_f with some random flips
                f = best_f.copy()
                for _ in range(10):
                    idx = np.random.randint(N)
                    f[idx] *= -1
                current_score = _calc_metric(f)
                no_improvement_count = 0
        else:
            no_improvement_count = 0

    return best_f

# EVOLVE-BLOCK-END
