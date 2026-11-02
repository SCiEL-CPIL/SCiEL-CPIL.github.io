"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(f):
    """Compute full linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    return ac


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)


def construct():
    deadline = time.time() + 120.0  # safety margin before 150s kill
    
    # Start with all ones (the seed)
    best = np.ones(N)
    best_score = _score(best)
    
    # Try to improve via local search on continuous values, then round
    # Since the ratio is scale-invariant, fix sum = N (i.e., mean = 1)
    
    # Local search: perturb individual elements
    f = best.copy().astype(float)
    current_score = best_score
    
    step_size = 0.5
    np.random.seed(42)
    
    for iteration in range(50000):
        if time.time() > deadline - 10.0:
            break
        
        # Pick a random index
        idx = np.random.randint(N)
        old_val = f[idx]
        f[idx] += np.random.choice([-1, 1]) * step_size
        
        # Renormalize to keep sum = N (scale-invariant, so this is fine)
        s = np.sum(f)
        if abs(s) < 1e-10:
            f[idx] = old_val
            continue
        f = f * (N / s)
        
        sc = _score(f)
        if sc < current_score:
            current_score = sc
            if current_score < best_score:
                best_score = current_score
                best = f.copy()
        else:
            # Revert
            f[idx] = old_val
            s = np.sum(f)
            if abs(s) >= 1e-10:
                f = f * (N / s)
            else:
                f[idx] = old_val
                f = f * (N / max(abs(s), 1e-10))
    
    # Round to signed values: sign of each element
    result = np.sign(best)
    # Handle zeros
    result[result == 0] = 1.0
    
    # Check if rounded result is better
    rounded_score = _score(result)
    if rounded_score < best_score:
        return result
    
    # Try a few random ±1 sequences
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        rand_seq = np.random.choice([-1.0, 1.0], N)
        rs = _score(rand_seq)
        if rs < best_score:
            best_score = rs
            best = rand_seq
    
    return np.sign(best)
# EVOLVE-BLOCK-END
