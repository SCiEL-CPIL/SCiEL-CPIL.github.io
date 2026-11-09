"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_max_abs_excl_lag0(f):
    """Compute max |autocorr(k)| for k != 0 using FFT."""
    n = len(f)
    # Full autocorrelation via FFT
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * np.conj(F), size)
    # ac[0] is lag 0, ac[1:n] are positive lags, ac[size-n+1:size] are negative lags
    # For real signals, ac is symmetric: ac[k] = ac[size-k]
    # We want max |ac[k]| for k = 1, ..., n-1
    # Positive lags: ac[1:n]
    # Negative lags: ac[size-n+1:size] (which equals ac[n-1:0:-1])
    pos = ac[1:n]
    neg = ac[size - n + 1:size]
    return np.maximum(np.max(np.abs(pos)), np.max(np.abs(neg)))

def _objective(f):
    """Compute the score: max|autocorr(k)| for k!=0 divided by (sum(f))^2."""
    S = np.sum(f)
    if S == 0:
        return np.inf
    max_ac = _autocorr_max_abs_excl_lag0(f)
    return max_ac / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    # Initialize with a ±1 sequence
    # Start with a structured pattern that might have low autocorrelation
    # Try a few initial patterns
    best_f = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several initializations
    inits = []
    # Alternating
    inits.append(np.array([1.0, -1.0] * (N // 2)))
    # Random
    inits.append(rng.choice([-1.0, 1.0], size=N))
    # Block pattern
    inits.append(np.tile([1.0, 1.0, -1.0, -1.0], N // 4))
    # Based on quadratic residue (for prime-like N)
    # Use a simple construction
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    for f0 in inits:
        if time.time() > deadline:
            break
        f = f0.copy()
        score = _objective(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    if best_f is None:
        return np.ones(N)
    
    f = best_f.copy()
    current_score = best_score
    
    # Local search: try flipping individual elements
    # Use a greedy approach with random order
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Random order of indices
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            # Try flipping element i
            old_val = f[i]
            f[i] = -old_val
            new_score = _objective(f)
            if new_score < current_score:
                current_score = new_score
                best_f = f.copy()
                best_score = new_score
                improved = True
            else:
                f[i] = old_val
    
    # Second pass: try pairs of flips for more diverse search
    if time.time() < deadline - 5:
        f = best_f.copy()
        current_score = best_score
        for _ in range(2000):
            if time.time() > deadline:
                break
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            f[i] = -f[i]
            f[j] = -f[j]
            new_score = _objective(f)
            if new_score < current_score:
                current_score = new_score
                best_f = f.copy()
                best_score = new_score
            else:
                f[i] = -f[i]
                f[j] = -f[j]
    
    return best_f
# EVOLVE-BLOCK-END
