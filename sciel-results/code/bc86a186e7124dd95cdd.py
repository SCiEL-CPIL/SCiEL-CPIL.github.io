"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(f):
    """Compute max |autocorrelation| for lags 1..N-1, normalized by N."""
    n = len(f)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    # Exclude lag 0, take max absolute value of remaining lags
    if n > 1:
        return np.max(np.abs(acf[1:])) / n
    return 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Initialize with random signs
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _max_sidelobe(best_f)
    
    # Local search with multiple restarts
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start from best found so far or a new random initialization
        if restart == 0:
            current_f = best_f.copy()
            current_score = best_score
        else:
            current_f = rng.choice([-1.0, 1.0], size=N)
            current_score = _max_sidelobe(current_f)
            
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
        
        # Greedy local search: try flipping each position in random order
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            order = rng.permutation(N)
            for i in order:
                if time.time() > deadline:
                    break
                
                # Flip sign at position i
                current_f[i] *= -1.0
                new_score = _max_sidelobe(current_f)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score - 1e-12:
                        best_score = current_score
                        best_f = current_f.copy()
                else:
                    # Revert the flip
                    current_f[i] *= -1.0
    
    return best_f
# EVOLVE-BLOCK-END
