"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(f):
    """Compute max|autoconv| (excluding lag 0) / (sum(f))²"""
    s = f.sum()
    if abs(s) < 1e-10:
        return float('inf')
    # Circular autoconvolution via FFT
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # ac[0] is lag 0, exclude it
    max_abs = np.max(np.abs(ac[1:]))
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Start with all ones
    f = np.ones(N, dtype=np.float64)
    best_score = _compute_score(f)
    best_f = f.copy()
    
    # Precompute autoconvolution
    def get_autoconv(f):
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        return np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    
    ac = get_autoconv(f)
    s = f.sum()
    
    # Local search: flip signs one at a time
    for iteration in range(500000):
        if time.time() > deadline - 5.0:
            break
        
        best_gain = 0
        best_flip = -1
        
        for i in range(N):
            # Trial flip
            old_val = f[i]
            f[i] = -f[i]
            
            new_s = s - 2 * old_val
            if abs(new_s) < 1e-10:
                f[i] = old_val
                continue
            
            # Update autoconvolution incrementally
            # ac[k] = sum_j f[j]*f[j+k] (circular)
            # Flipping f[i] changes ac[k] by -2*f[i]*f[(i+k)%N] for each k
            # This is O(N) per trial, too slow for 400*500000
            
            # Just compute directly for now
            new_ac = get_autoconv(f)
            new_max = np.max(np.abs(new_ac[1:]))
            new_score = new_max / (new_s * new_s)
            
            if new_score < best_score - 1e-12:
                best_gain = best_score - new_score
                best_flip = i
                best_new_s = new_s
                best_new_ac = new_ac.copy()
            else:
                f[i] = old_val
        
        if best_flip == -1:
            # No improvement, try random restarts
            if iteration % 1000 == 0 and time.time() < deadline - 10.0:
                # Random restart with bias toward +1
                f = np.ones(N, dtype=np.float64)
                num_flips = np.random.randint(10, 50)
                flip_idx = np.random.choice(N, num_flips, replace=False)
                f[flip_idx] = -1
                s = f.sum()
                ac = get_autoconv(f)
                best_score = _compute_score(f)
                best_f = f.copy()
            else:
                break
        else:
            f[best_flip] = -f[best_flip]
            s = best_new_s
            ac = best_new_ac
            best_score = _compute_score(f)
            best_f = f.copy()
    
    return best_f
# EVOLVE-BLOCK-END
