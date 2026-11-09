"""
Optimized search for a signed step function minimizing the peak aperiodic 
autocorrelation sidelobe.

The task asks to improve an autoconvolution upper bound (C3). 
We interpret this as minimizing the peak magnitude of the non-zero lags 
of the aperiodic autocorrelation.

The seed program returns np.ones(N), which has a peak sidelobe of N-1 = 399.
Random sequences typically have peak sidelobes around sqrt(N) ~ 20.
We use a combination of random initialization and greedy local search 
with occasional multi-bit flips to escape local minima.
"""

import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin before 120s kill

def compute_peak_sidelobe(h):
    """
    Compute the peak magnitude of the aperiodic autocorrelation 
    for non-zero lags (1 to N-1).
    """
    # Zero-pad to 2*N to ensure linear (aperiodic) autocorrelation
    n_fft = 2 * N
    # Use rfft for real input
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0. r[1:N] are lags 1..N-1.
    # Due to symmetry, negative lags are the same magnitude.
    # We only need to check 1..N-1.
    if N > 1:
        return np.max(np.abs(r[1:N]))
    else:
        return 0.0

def construct():
    """
    Construct a signed step function (sequence of +/- 1) that minimizes 
    the peak aperiodic autocorrelation sidelobe.
    """
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Random Initialization
    # A random sequence is already a significant improvement over np.ones
    h = rng.choice([-1.0, 1.0], size=N)
    current_peak = compute_peak_sidelobe(h)
    best_h = h.copy()
    best_peak = current_peak
    
    # Phase 2: Greedy Local Search with Multi-bit Flips
    # We alternate between single-bit flips (fine polish) and 
    # block flips (escape local minima) in a simulated annealing-like manner
    # but with a focus on fast convergence.
    
    T = 10.0  # Initial "temperature" for accepting worse solutions
    T_min = 0.1
    cooling_rate = 0.999
    
    # We will run until the deadline is approaching
    while time.time() < deadline - 5.0:  # Leave 5s for final polish
        # Determine flip size: mostly 1, occasionally larger blocks
        # This helps escape local minima in the binary hypercube
        if rng.random() < 0.1:
            flip_size = rng.integers(2, 5)  # Flip 2-4 bits
        else:
            flip_size = 1
        
        indices = rng.choice(N, size=flip_size, replace=False)
        
        # Save old values
        old_values = h[indices].copy()
        
        # Flip
        h[indices] = -h[indices]
        
        new_peak = compute_peak_sidelobe(h)
        delta = new_peak - current_peak
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_h = h.copy()
        else:
            # Revert
            h[indices] = old_values
        
        T *= cooling_rate
        
        # If we've hit a very good solution, tighten the search
        if best_peak < 22.0:
            # Switch to pure greedy single-bit polish for the remaining time
            break
    
    # Phase 3: Final Greedy Polish (Single-bit flips only)
    # This ensures we are at a local minimum for single-bit flips
    if time.time() < deadline - 1.0:
        h = best_h.copy()
        current_peak = best_peak
        polish_deadline = deadline - 1.0
        
        improved = True
        # Limit iterations to avoid infinite loop if time is tight
        max_iterations = 100 
        iterations = 0
        
        while improved and time.time() < polish_deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                
                old_val = h[i]
                h[i] = -old_val
                
                new_peak = compute_peak_sidelobe(h)
                
                if new_peak < current_peak - 1e-12:
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_h = h.copy()
                else:
                    # Revert
                    h[i] = old_val
    
    return best_h
