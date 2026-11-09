"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_max_abs(x):
    """Compute max absolute autocorrelation excluding lag 0."""
    n = len(x)
    # Use FFT-based autocorrelation for speed
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # Exclude lag 0
    return np.max(np.abs(acf[1:]))

def _score(x):
    """Score: max|autocorr(k)| / (sum(x))^2, lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return float('inf')
    return _autocorr_max_abs(x) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s limit
    
    # Start with random ±1 sequence
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N).astype(np.float64)
    
    # Ensure sum is not zero
    # If sum is 0, flip one element
    while abs(np.sum(x)) < 1.0:
        idx = rng.randint(N)
        x[idx] *= -1
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Local search: try flipping individual signs
    # Precompute which flips help
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try all single flips
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            old_val = x[i]
            x[i] = -old_val
            
            s = np.sum(x)
            if abs(s) < 1e-10:
                x[i] = old_val
                continue
            
            acf = _autocorr_max_abs(x)
            new_score = acf / (s * s)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
        
        # If no improvement with single flips, try random restarts
        if not improved:
            # Try a few random restarts
            for restart in range(5):
                if (time.time() - start_time) > deadline:
                    break
                x_new = rng.choice([-1, 1], size=N).astype(np.float64)
                while abs(np.sum(x_new)) < 1.0:
                    idx = rng.randint(N)
                    x_new[idx] *= -1
                s = np.sum(x_new)
                acf = _autocorr_max_abs(x_new)
                sc = acf / (s * s)
                if sc < best_score:
                    best_score = sc
                    best_x = x_new.copy()
                    x = x_new.copy()
                    improved = True
                    break
    
    return best_x
# EVOLVE-BLOCK-END
