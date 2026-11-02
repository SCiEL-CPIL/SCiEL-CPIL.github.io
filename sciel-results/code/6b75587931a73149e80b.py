import numpy as np
import time

N = 600

def _score(f):
    """Compute the normalized maximum of autoconvolution."""
    f = np.asarray(f, dtype=float)
    s = np.sum(f)
    if s == 0:
        return 0.0
    # Autoconvolution via FFT
    size = 1 << (2 * len(f) - 1).bit_length()
    F = np.fft.rfft(f, n=size)
    ac = np.fft.irfft(F * np.conj(F), n=size)[:len(f)]
    # Try different normalizations
    # Option 1: max(ac) / ac[0]
    # Option 2: max(ac) / (s**2)
    # Option 3: max(ac) / s
    # The task says "normalized maximum" - likely max(ac)/(sum(f))^2 or similar
    return float(np.max(ac) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Random binary
    np.random.seed(123)
    for _ in range(20):
        candidates.append((np.random.rand(N) < 0.5).astype(float))
    
    # Alternating
    candidates.append((np.arange(N) % 2).astype(float))
    
    # Block patterns
    for block in [2, 3, 5, 7, 10]:
        candidates.append((np.arange(N) // block) % 2)
    
    best_f = candidates[0]
    best_score = _score(best_f)
    
    for f in candidates[1:]:
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f
    
    # Try local search on best
    T = 0.1
    for i in range(100000):
        if time.time() > deadline:
            break
        idx = np.random.randint(N)
        old = best_f[idx]
        best_f[idx] = 1.0 - old
        new_sc = _score(best_f)
        if new_sc < best_score:
            best_score = new_sc
        else:
            best_f[idx] = old
        T *= 0.9999
    
    return best_f.tolist()
