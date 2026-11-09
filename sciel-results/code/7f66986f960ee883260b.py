import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Initialize with random ±1 sequence
    rng = np.random.default_rng(42)
    a = rng.choice([-1, 1], size=N)
    
    def max_autocorr(seq):
        """Compute max absolute autocorrelation."""
        # R(k) = sum_{i=0}^{N-1-k} seq[i]*seq[i+k]
        # Use FFT for efficiency
        n = len(seq)
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:n]
        # ac[0] is the energy, we want lags 1..n-1
        return np.max(np.abs(ac[1:]))
    
    def autocorr_sum_sq(seq):
        """Sum of squared autocorrelations for lags 1..n-1."""
        n = len(seq)
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:n]
        return np.sum(ac[1:]**2)
    
    # We'll minimize max autocorrelation
    best_a = a.copy()
    best_score = max_autocorr(a)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Try random bit flips
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = rng.integers(N)
            a[idx] *= -1
            new_score = max_autocorr(a)
            delta = new_score - best_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                best_a = a.copy()
                best_score = new_score
            else:
                a[idx] *= -1  # revert
        T *= cooling
    
    # Local search with single flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            best_a[idx] *= -1
            new_score = max_autocorr(best_a)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_a[idx] *= -1
    
    return best_a
