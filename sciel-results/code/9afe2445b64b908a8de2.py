"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_max_abs(f):
    """Compute max absolute value of full autoconvolution (linear, not circular)."""
    # Use FFT for efficiency
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * np.conj(F), size)
    # The full linear autocorrelation is in ac[0:2*n-1]
    # But irfft gives us the circular result. For linear autocorrelation:
    # ac[0] = sum(f^2), ac[k] for k>=1 is the positive lags
    # The negative lags are the same as positive for real signals
    return np.max(np.abs(ac[:2*n-1]))


def _objective(f):
    """Score: max_abs_autoconv / (sum(f))^2. Lower is better."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return _autocorr_max_abs(f) / (s * s)


def _try_local_search(f, max_time=100.0):
    """Local search: flip individual elements to minimize objective."""
    start_time = time.time()
    best_f = f.copy()
    best_score = _objective(best_f)
    n = len(best_f)
    
    # Precompute some useful quantities
    # For flipping element i from +1 to -1 or vice versa:
    # new_sum = old_sum - 2*f[i]
    # We need to recompute the max autocorrelation efficiently
    
    # Use a simple approach: try all single flips, pick the best improvement
    # Repeat until no improvement or time runs out
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > max_time - 5:
            break
        
        # Try flipping each element
        best_improvement = 0
        best_idx = -1
        old_sum = np.sum(best_f)
        
        for i in range(n):
            # Flip element i
            old_val = best_f[i]
            best_f[i] = -old_val
            
            new_sum = old_sum - 2 * old_val
            if abs(new_sum) < 1e-12:
                best_f[i] = old_val
                continue
            
            score = _autocorr_max_abs(best_f) / (new_sum * new_sum)
            best_f[i] = old_val
            
            if score < best_score - 1e-12:
                best_improvement = best_score - score
                best_idx = i
        
        if best_idx >= 0:
            best_f[best_idx] = -best_f[best_idx]
            best_score = _objective(best_f)
            improved = True
    
    return best_f, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start from all ones
    f = np.ones(N)
    best_score = _objective(f)
    
    # Phase 1: Random initialization with random signs
    # Try several random sign patterns
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        f_rand = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(f_rand)
        if abs(s) < 1e-12:
            continue
        score = _autocorr_max_abs(f_rand) / (s * s)
        if score < best_score:
            best_score = score
            f = f_rand.copy()
    
    # Phase 2: Local search from best starting point
    f, best_score = _try_local_search(f, max_time=90.0)
    
    # Phase 3: Try more random starts if time remains
    if time.time() < deadline - 30:
        for trial in range(10):
            if time.time() > deadline - 15:
                break
            f_rand = rng.choice([-1.0, 1.0], size=N)
            s = np.sum(f_rand)
            if abs(s) < 1e-12:
                continue
            score = _autocorr_max_abs(f_rand) / (s * s)
            if score < best_score:
                best_score = score
                f = f_rand.copy()
        
        # Local search from new best if it improved
        if time.time() < deadline - 10:
            f, new_score = _try_local_search(f, max_time=20.0)
            if new_score < best_score:
                best_score = new_score
    
    return f
# EVOLVE-BLOCK-END
