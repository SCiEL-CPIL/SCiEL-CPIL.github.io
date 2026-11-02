"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_bound(x):
    """Compute the upper bound on C3 for a signed step function x.
    
    The autoconvolution r(k) = sum_i x_i * x_{i+k} for lag k.
    The 'upper bound on C3' relates to the maximum off-peak autocorrelation.
    We minimize the max |r(k)| for k != 0.
    """
    n = len(x)
    # Use FFT for fast autocorrelation
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # acf[0] is the peak (sum of squares), ignore it
    # Off-peak values
    off_peak = np.concatenate([acf[1:], acf[-1:][::-1]])  # this isn't quite right for one-sided
    # Actually for real autocorrelation, r(k) = r(-k), so we just look at k=1..n-1
    off_peak = acf[1:]
    return np.max(np.abs(off_peak))

def _evaluate_flip(x, idx, current_bound):
    """Evaluate what happens if we flip x[idx]. Returns new bound value."""
    # Flipping x[idx] changes autocorrelation
    # r_new(k) = r_old(k) - 2*x[idx]*x[idx+k] for k>0 (when idx+k < n)
    #           r_new(k) = r_old(k) - 2*x[idx]*x[idx-k] for k>0 (when idx-k >= 0)
    #           r_new(k) = r_old(k) + 2*x[idx]*x[idx+k] ... wait
    # Actually: r(k) = sum_i x[i]*x[i+k]
    # If we flip x[idx] -> -x[idx]:
    #   Terms where i=idx: x[idx]*x[idx+k] -> -x[idx]*x[idx+k], change = -2*x[idx]*x[idx+k]
    #   Terms where i+k=idx, i.e., i=idx-k: x[idx-k]*x[idx] -> x[idx-k]*(-x[idx]), change = -2*x[idx-k]*x[idx]
    # So r_new(k) = r_old(k) - 2*x[idx]*x[idx+k] - 2*x[idx-k]*x[idx] (with bounds checking)
    
    # For efficiency, let's just recompute via FFT if n is small enough
    # But for n=400, direct computation per flip is O(n) = 400 ops, which is fine
    # Let's do it directly for clarity and correctness
    
    x_new = x.copy()
    x_new[idx] = -x_new[idx]
    
    # Compute autocorrelation directly
    r = np.zeros(N)
    for k in range(N):
        if k == 0:
            r[k] = np.sum(x_new * x_new)
        else:
            r[k] = np.sum(x_new[:N-k] * x_new[k:])
    
    off_peak = r[1:]
    return np.max(np.abs(off_peak))

def _evaluate_flip_fast(x, idx, acf):
    """Fast evaluation of flipping x[idx] using precomputed autocorrelation.
    
    acf[k] = sum_i x[i]*x[i+k] for k=0..N-1
    When we flip x[idx] to -x[idx]:
      For lag k > 0:
        - Term i=idx: was x[idx]*x[idx+k], now -x[idx]*x[idx+k], delta = -2*x[idx]*x[idx+k] (if idx+k < N)
        - Term i=idx-k: was x[idx-k]*x[idx], now x[idx-k]*(-x[idx]), delta = -2*x[idx-k]*x[idx] (if idx-k >= 0)
      So acf_new[k] = acf[k] - 2*x[idx]*x[idx+k] - 2*x[idx-k]*x[idx]
      
    For k=0: acf_new[0] = acf[0] (unchanged since x[idx]^2 is same)
    """
    x_val = x[idx]
    acf_new = acf.copy()
    
    for k in range(1, N):
        delta = 0.0
        if idx + k < N:
            delta -= 2.0 * x_val * x[idx + k]
        if idx - k >= 0:
            delta -= 2.0 * x[idx - k] * x_val
        acf_new[k] = acf[k] + delta
    
    off_peak = acf_new[1:]
    return np.max(np.abs(off_peak))

def _compute_acf(x):
    """Compute autocorrelation r[k] = sum_i x[i]*x[i+k] for k=0..N-1."""
    n = len(x)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.sum(x[:n-k] * x[k:])
    return r

def construct():
    deadline = time.time() + 140  # 140s to leave safety margin
    
    # Start with random initialization
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial autocorrelation
    acf = _compute_acf(x)
    current_bound = np.max(np.abs(acf[1:]))
    
    best_bound = current_bound
    best_x = x.copy()
    
    # Greedy coordinate descent
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_flip_bound = current_bound
        
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            new_bound = _evaluate_flip_fast(x, idx, acf)
            if new_bound < best_flip_bound - 1e-12:
                best_flip_bound = new_bound
                best_flip_idx = idx
        
        if best_flip_idx >= 0:
            # Apply the best flip
            x[best_flip_idx] = -x[best_flip_idx]
            acf = _compute_acf(x)
            current_bound = np.max(np.abs(acf[1:]))
            improved = True
            
            if current_bound < best_bound:
                best_bound = current_bound
                best_x = x.copy()
    
    # If we have time, try a few random restarts
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            x = np.random.choice([-1.0, 1.0], size=N)
            acf = _compute_acf(x)
            current_bound = np.max(np.abs(acf[1:]))
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                best_flip_idx = -1
                best_flip_bound = current_bound
                
                for idx in range(N):
                    if time.time() > deadline - 5:
                        break
                    new_bound = _evaluate_flip_fast(x, idx, acf)
                    if new_bound < best_flip_bound - 1e-12:
                        best_flip_bound = new_bound
                        best_flip_idx = idx
                
                if best_flip_idx >= 0:
                    x[best_flip_idx] = -x[best_flip_idx]
                    acf = _compute_acf(x)
                    current_bound = np.max(np.abs(acf[1:]))
                    improved = True
                    
                    if current_bound < best_bound:
                        best_bound = current_bound
                        best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
