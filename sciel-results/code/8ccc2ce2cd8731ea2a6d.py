"""Rewritten program: heuristic search for signed step heights minimizing max|autoconv|/(sum)^2."""
import numpy as np
import time

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    # Use next power of 2 for efficiency
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _score(x):
    """Compute the metric: max|c_k| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    return np.max(np.abs(c)) / (s * s)

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 110s safety margin
    
    best_x = np.ones(N)
    best_val = _score(best_x)
    
    # Try various structured sequences
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. All ones with small perturbations
    rng = np.random.default_rng(42)
    
    # 3. Try sequences based on chirps, windows, etc.
    # Hanning window (positive, large sum)
    candidates.append(np.hanning(N))
    candidates.append(np.hanning(N) - 0.5)  # centered
    
    # 4. Try to reduce peak autocorr: use a sequence with low sidelobes
    # Barker-like or Zadoff-Chu inspired
    # Zadoff-Chu sequence
    root = 7  # relatively prime to N? N=400, gcd(7,400)=1
    zc = np.exp(-2j * np.pi * root * np.arange(N) * np.arange(N) / N)
    # Take real part
    candidates.append(np.real(zc))
    candidates.append(np.imag(zc))
    
    # 5. Random sequences with various properties
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        # Random with bias toward positive
        r = rng.random(N)
        x = np.where(r > 0.3, 1.0, -1.0)
        candidates.append(x)
    
    # 6. Try sequences with a large constant offset + small variation
    for offset in [1.0, 2.0, 5.0, 10.0]:
        if time.time() > deadline - 5:
            break
        noise = rng.normal(0, 0.1, N)
        candidates.append(offset + noise)
    
    # 7. Try a "peaky" autocorrelation design: start with random, then optimize
    # Simple local search on a few candidates
    for start_idx, x0 in enumerate(candidates[:5]):
        if time.time() > deadline - 10:
            break
        
        x = x0.copy()
        cur_val = _score(x)
        
        # Coordinate descent: try flipping individual elements
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try a random subset of positions to flip
            idxs = rng.choice(N, size=min(20, N), replace=False)
            for idx in idxs:
                if time.time() > deadline - 5:
                    break
                old = x[idx]
                # Try a few new values
                for new_val in [old * 0.5, old * 1.5, -old, old + 0.5, old - 0.5]:
                    x[idx] = new_val
                    v = _score(x)
                    if v < cur_val - 1e-12:
                        cur_val = v
                        improved = True
                    else:
                        x[idx] = old
            
            if not improved:
                break
        
        if cur_val < best_val:
            best_val = cur_val
            best_x = x.copy()
    
    return best_x
