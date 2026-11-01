"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(s):
    """Compute autocorrelation of ±1 sequence using FFT."""
    n = len(s)
    # Zero-pad to next power of 2 for efficient FFT
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.zeros(size, dtype=np.float64)
    f[:n] = s
    F = np.fft.rfft(f)
    ac = np.fft.irfft(F * np.conj(F), n=size)[:n]
    return ac

def _max_sidelobe(s):
    """Return max absolute autocorrelation at lags 1..N-1."""
    ac = _autocorr(s)
    return np.max(np.abs(ac[1:]))

def _flip_cost(s, i):
    """Estimate cost change if we flip position i."""
    # Flipping s[i] changes autocorrelation at lag k by -2*s[i]*s[i+k] or -2*s[i]*s[i-k]
    # For efficiency, compute full autocorrelation change
    n = len(s)
    size = 1
    while size < 2 * n:
        size *= 2
    
    # Current autocorrelation
    f = np.zeros(size, dtype=np.float64)
    f[:n] = s
    F = np.fft.rfft(f)
    ac = np.fft.irfft(F * np.conj(F), n=size)[:n]
    
    # Flipped
    s2 = s.copy()
    s2[i] = -s2[i]
    f2 = np.zeros(size, dtype=np.float64)
    f2[:n] = s2
    F2 = np.fft.rfft(f2)
    ac2 = np.fft.irfft(F2 * np.conj(F2), n=size)[:n]
    
    return np.max(np.abs(ac2[1:]))

def construct():
    deadline = time.time() + 140  # Safety margin
    
    np.random.seed(42)
    best_s = np.random.choice([-1.0, 1.0], size=N)
    best_cost = _max_sidelobe(best_s)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    current_s = best_s.copy()
    current_cost = best_cost
    
    while time.time() < deadline and T > T_min:
        # Try a random flip
        i = np.random.randint(N)
        current_s[i] *= -1.0
        new_cost = _max_sidelobe(current_s)
        
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_s = current_s.copy()
        else:
            current_s[i] *= -1.0  # revert
        
        T *= cooling
    
    return best_s
# EVOLVE-BLOCK-END
