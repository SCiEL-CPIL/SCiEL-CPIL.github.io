# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _peak_sidelobe(f):
    """Compute the peak sidelobe level of the autocorrelation."""
    # Linear autocorrelation via FFT
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    # R[0] is the DC term (sum of squares = N)
    # R[1:N] are positive lags
    # R[N:2N-1] are negative lags (symmetric)
    # We want max |R[k]| for k != 0
    # Due to symmetry, max(R[1:N], R[N+1:2N]) is sufficient, but R[N+1:2N] corresponds to negative lags
    # Actually, for real signals, R[k] = R[-k].
    # The FFT output R[0..2N-1] contains lags 0..N-1 and -N..-1.
    # Specifically, R[k] for k=0..N-1 is the linear autocorrelation for lag k.
    # R[k] for k=N..2N-1 is the linear autocorrelation for lag k-N (negative).
    # So we just need to check R[1:N] and R[N+1:2N] (excluding R[N] which is lag 0? No, R[N] is lag 0 in circular? No.)
    # Let's stick to the first N elements for positive lags 0..N-1.
    # The max sidelobe is max(|R[1:N]|, |R[N+1:2N]|).
    # But R[N+1:2N] is the same as R[N-1:0:-1].
    # So just checking R[1:N] and R[N+1:2N] covers all non-zero lags.
    # Actually, R[0] is lag 0.
    # Lags 1 to N-1 are in R[1:N].
    # Lags -1 to -(N-1) are in R[N+1:2N] (reversed).
    # So max_abs_sidelobe = max(np.max(np.abs(R[1:N])), np.max(np.abs(R[N+1:2N])))
    # But R[N] is the value at lag N? For zero-padded FFT, R[N] is the correlation at lag N, which is 0 for linear?
    # Let's verify.
    # x = [1, 1, 0, 0] (N=2, pad to 4).
    # R = [2, 1, 0, 1].
    # R[0]=2 (lag 0). R[1]=1 (lag 1). R[2]=0 (lag 2). R[3]=1 (lag -1).
    # So R[N] is lag N, which is 0.
    # So we can just take max(abs(R[1:2N]))?
    # R[1] to R[N-1] are positive lags.
    # R[N] is 0.
    # R[N+1] to R[2N-1] are negative lags.
    # So yes, max(abs(R[1:2N])) works.
    
    if N == 1:
        return 0.0
    return np.max(np.abs(R[1:2*N]))

def _compute_score(f):
    """Internal score for optimization: Peak Sidelobe."""
    return _peak_sidelobe(f)

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize with random signs
    np.random.seed(42)
    f = np.random.choice([-1.0, 1.0], size=N)
    current_val = _compute_score(f)
    best_f = f.copy()
    best_val = current_val
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    # Pre-compute FFT size
    n_fft = 2 * N
    
    while (time.time() - start_time) < time_limit:
        # Perform a batch of moves
        for _ in range(100):
            if (time.time() - start_time) > time_limit:
                break
            
            # Flip a random element
            idx = np.random.randint(N)
            f[idx] *= -1.0
            
            new_val = _compute_score(f)
            delta = new_val - current_val
            
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_f = f.copy()
            else:
                # Revert
                f[idx] *= -1.0
        
        T *= alpha
        if T < T_min:
            break
    
    # Greedy Polish
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try single flips
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
            f[i] *= -1.0
            new_val = _compute_score(f)
            if new_val < current_val - 1e-15:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_f = f.copy()
            else:
                f[i] *= -1.0
    
    return best_f
# EVOLVE-BLOCK-END
