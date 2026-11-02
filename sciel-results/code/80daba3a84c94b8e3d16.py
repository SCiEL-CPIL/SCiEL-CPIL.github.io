# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(s.astype(np.float64), n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] is sum s[i]s[i+k]
    # We exclude lag 0
    return float(np.max(np.abs(acf[1:])))

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initialize with random sequence
    s = np.random.choice([-1.0, 1.0], size=N)
    
    # Optionally try a few different random starts or structured starts
    # A simple structured start: alternating or random. Random is usually fine.
    
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    
    # Local Search
    # To speed up, we can precompute the effect of a flip?
    # Flipping s[i] from v to -v changes acf[k] by -2*v*s[i+k] for k>0 and -2*v*s[i-k] for k<0 (if we considered negative lags, but acf is symmetric for real signals, so we just look at positive lags).
    # Actually, R(k) = sum_{j=0}^{N-1-k} s[j]s[j+k].
    # If we flip s[i], the new R(k) differs from old R(k) by:
    # - 2*s[i]*s[i+k] if i+k < N (term s[i]s[i+k] becomes -s[i]s[i+k])
    # - 2*s[i]*s[i-k] if i-k >= 0 (term s[i-k]s[i] becomes -s[i-k]s[i])
    # Note: R(k) for k>0.
    
    # However, doing the full FFT for each flip is simpler and fast enough for N=400.
    # 400 flips * 1000 passes = 400,000 FFTs.
    # One FFT of size 1024 is ~10us? 400,000 * 10us = 4 seconds.
    # This is well within the limit.
    
    while time.time() < deadline:
        improved = False
        # Iterate through indices in random order to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            # Flip s[i]
            s[i] *= -1
            
            new_val = _autocorr_max_abs(s)
            
            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_s = s.copy()
                improved = True
            else:
                # Revert
                s[i] *= -1
        
        if not improved:
            # Try a few random restarts if stuck? Or just break.
            # Random restarts can escape local minima.
            s = np.random.choice([-1.0, 1.0], size=N)
            current_val = _autocorr_max_abs(s)
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
            # Check time again
            if time.time() > deadline:
                break

    return best_s
# EVOLVE-BLOCK-END
