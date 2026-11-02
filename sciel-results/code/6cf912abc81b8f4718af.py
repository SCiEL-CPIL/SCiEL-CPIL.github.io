"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _acf_max_abs(f):
    """Compute max |ACF(k)| for k=1..N-1."""
    n = len(f)
    # Use FFT for efficient autocorrelation
    f_ext = np.zeros(2 * n, dtype=float)
    f_ext[:n] = f
    F = np.fft.rfft(f_ext, n=2 * n)
    acf_full = np.fft.irfft(F * np.conj(F), n=2 * n)[:n]
    # acf_full[0] is lag 0, acf_full[1..n-1] are lags 1..n-1
    # Also need negative lags? For real signals, ACF is symmetric.
    # But our step function on a finite interval: ACF(k) = sum_{i} f(i)*f(i+k) for k>0
    # and ACF(-k) = sum_{i} f(i)*f(i-k) = sum_{j} f(j+k)*f(j) = ACF(k)
    # So max over k=1..n-1 of |ACF(k)| suffices.
    return np.max(np.abs(acf_full[1:]))


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    rng = np.random.default_rng(42)
    
    # Initialize with random ±1 sequence
    best = rng.choice([-1.0, 1.0], size=N)
    best_score = _acf_max_abs(best)
    
    current = best.copy()
    current_score = best_score
    
    # Simulated annealing with block-flip moves
    T = 50.0  # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Choose block size
        block_size = rng.integers(2, 33)  # 2 to 32
        if block_size > N:
            block_size = N
        start_idx = rng.integers(0, N - block_size + 1)
        
        # Create candidate
        candidate = current.copy()
        candidate[start_idx:start_idx + block_size] *= -1.0
        
        candidate_score = _acf_max_abs(candidate)
        
        delta = candidate_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = candidate
            current_score = candidate_score
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    return best
# EVOLVE-BLOCK-END
