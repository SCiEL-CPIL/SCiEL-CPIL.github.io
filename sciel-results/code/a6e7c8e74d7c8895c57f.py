"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(a):
    """Compute max absolute autocorrelation for lags 1..N-1 using FFT."""
    n = len(a)
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * n:
        size *= 2
    fa = np.fft.rfft(a, size)
    ac = np.fft.irfft(fa * np.conj(fa), size)[:n]
    # ac[0] = n (sum of squares), we want max of |ac[1:]|
    if n > 1:
        return np.max(np.abs(ac[1:]))
    return 0.0


def _flip_bit(a, i):
    """Flip bit i and return new peak autocorrelation efficiently."""
    # For efficiency, just recompute - but we can use the structure
    # Actually for N=400, FFT is fast enough
    a[i] = -a[i]
    peak = _autocorr_peak(a)
    a[i] = -a[i]  # undo
    return peak


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_a = None
    best_peak = np.inf
    
    # Try multiple random restarts with simulated annealing
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with random ±1 sequence
        a = np.random.choice([-1, 1], size=N)
        current_peak = _autocorr_peak(a)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_a = a.copy()
        
        # Simulated annealing
        temp = 50.0
        cooling = 0.995
        no_improve = 0
        
        while temp > 0.1:
            if time.time() > deadline - 5.0:
                break
            
            # Try a random flip
            i = np.random.randint(N)
            a[i] = -a[i]
            new_peak = _autocorr_peak(a)
            
            delta = new_peak - current_peak
            
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_a = a.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                a[i] = -a[i]  # undo
                no_improve += 1
            
            temp *= cooling
            
            # Early stop if stuck too long
            if no_improve > 500:
                break
        
        # If we found a very good solution, stop early
        if best_peak <= 15:
            break
    
    return best_a if best_a is not None else np.ones(N)
# EVOLVE-BLOCK-END
