"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    # Use FFT to compute autocorrelation
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    # acf[0] is the DC term (sum of squares = N for +/-1)
    # We want max |acf[k]| for k = 1..n-1
    return np.max(np.abs(acf[1:]))


def _try_flip(h, i, deadline):
    """Try flipping bit i, return new max sidelobe if improved."""
    h[i] = -h[i]
    val = _max_sidelobe(h)
    h[i] = -h[i]
    return val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Start with a random +/-1 sequence
    np.random.seed(42)
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_val = _max_sidelobe(best_h)
    
    # Local search: greedy flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            h = best_h.copy()
            h[i] = -h[i]
            val = _max_sidelobe(h)
            if val < best_val:
                best_h = h
                best_val = val
                improved = True
    
    # If we have time, try simulated annealing from best
    if time.time() < deadline:
        h = best_h.copy()
        current_val = best_val
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline and T > 0.01:
            # Pick random position to flip
            i = np.random.randint(N)
            h[i] = -h[i]
            new_val = _max_sidelobe(h)
            
            delta = new_val - current_val
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_val = new_val
                if new_val < best_val:
                    best_h = h.copy()
                    best_val = new_val
            else:
                h[i] = -h[i]  # revert
            
            T *= cooling
    
    return best_h
# EVOLVE-BLOCK-END
