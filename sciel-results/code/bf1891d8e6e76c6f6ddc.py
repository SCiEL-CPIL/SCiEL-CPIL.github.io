"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _max_sidelobe(f):
    """Compute the maximum absolute autocorrelation value for non-zero lags, normalized by N."""
    n = len(f)
    # Next power of 2 >= 2n for efficient FFT
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    # Use rfft for real input
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    # acf[0] is the DC component (sum of squares), we want max |acf[k]| for k=1..n-1
    # Note: due to circular convolution in FFT, we only get positive lags 0..n-1 directly from irfft slice
    # However, for real signals, autocorrelation is symmetric.
    # The irfft output length is nfft//2 + 1. Slicing [:n] gives lags 0..n-1.
    # We need to exclude lag 0.
    return np.max(np.abs(acf[1:])) / n

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limit
    
    rng = np.random.default_rng(42)
    
    # Initialize with random signs
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _max_sidelobe(best_f)
    
    # Phase 1: Greedy Local Search with Random Restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        current_f = best_f.copy() if restart > 0 else rng.choice([-1.0, 1.0], size=N)
        current_score = _max_sidelobe(current_f)
        
        if current_score < best_score:
            best_score = current_score
            best_f = current_f.copy()
            
        improved = True
        # Local search loop
        while improved and (time.time() < deadline):
            improved = False
            # Try all flips in random order
            order = rng.permutation(N)
            for i in order:
                if time.time() > deadline:
                    break
                
                # Flip sign
                current_f[i] *= -1.0
                new_score = _max_sidelobe(current_f)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score - 1e-12:
                        best_score = current_score
                        best_f = current_f.copy()
                else:
                    # Revert
                    current_f[i] *= -1.0
        
        # If we hit the deadline, break out
        if time.time() > deadline:
            break
            
    # Phase 2: Simulated Annealing Polish
    # Try to escape local minima found by greedy search
    if time.time() < deadline - 10.0:
        current_f = best_f.copy()
        current_score = best_score
        T = 0.05  # Initial temperature
        cooling = 0.999
        min_T = 1e-4
        
        while T > min_T and time.time() < deadline - 5.0:
            idx = rng.integers(0, N)
            current_f[idx] *= -1.0
            new_score = _max_sidelobe(current_f)
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept: keep the flip
                current_score = new_score
                if current_score < best_score - 1e-12:
                    best_score = current_score
                    best_f = current_f.copy()
            else:
                # Reject: revert the flip
                current_f[idx] *= -1.0
            
            T *= cooling
            
    return best_f

# EVOLVE-BLOCK-END
