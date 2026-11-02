"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_sidelobes(a):
    """
    Compute the maximum absolute value of the autocorrelation 
    for non-zero lags.
    """
    # a is a 1D array of +/- 1
    # Autocorrelation c[k] = sum a[i] a[i+k]
    # We want max |c[k]| for k != 0
    # Using FFT for speed
    n = len(a)
    # Zero pad to 2n to avoid circular convolution effects
    size = 2 * n
    fa = np.fft.rfft(a, size)
    power = fa * np.conj(fa)
    c = np.fft.irfft(power, size)
    # c[0] is c(0) = sum a^2 = n
    # c[1..n-1] are positive lags
    # c[n+1..2n-1] correspond to negative lags (symmetric for real a)
    # Specifically, c[n-k] = c[k]
    # We only need to check k = 1 to n-1
    sidelobes = c[1:n]
    return np.max(np.abs(sidelobes))

def _score_val(a):
    """
    Calculate the raw value: max_abs_sidelobe / (sum a^2)^2
    Assuming sum a^2 = N for binary sequences.
    """
    max_side = _autocorr_sidelobes(a)
    integral_sq = (np.sum(a**2))**2
    return max_side / integral_sq

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Initialize with random signs
    # Using a seed for reproducibility in case of retries, but random is fine
    rng = np.random.default_rng(42)
    best_a = rng.choice([-1, 1], size=N)
    best_val = _score_val(best_a)
    
    # Local search: Flip bits to reduce max sidelobe
    # Heuristic: Find the lag with the largest sidelobe, 
    # and flip bits to reduce it.
    # Or simply: Random flips, accept if better.
    
    # To be more effective, we can use a greedy approach on the worst lag.
    # But random hill climbing with restarts is often robust.
    
    # Let's try a few restarts with different random seeds
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            a = best_a.copy()
        else:
            a = rng.choice([-1, 1], size=N)
        
        current_val = _score_val(a)
        
        # Hill Climbing
        no_improve_count = 0
        while time.time() < deadline - 5:
            if no_improve_count > 2000:
                break
            
            # Pick a random index to flip
            idx = rng.integers(0, N)
            old_val = a[idx]
            a[idx] = -old_val
            
            new_val = _score_val(a)
            
            if new_val < current_val:
                current_val = new_val
                no_improve_count = 0
                if current_val < best_val:
                    best_val = current_val
                    best_a = a.copy()
            else:
                a[idx] = old_val
                no_improve_count += 1
                
    return best_a
# EVOLVE-BLOCK-END
