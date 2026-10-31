import numpy as np
from scipy.signal import windows
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    # Use np.convolve for linear convolution
    ac = np.convolve(f, f, mode='full')
    # The peak is at index N-1
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_f = None
    best_val = float('inf')
    
    # Strategy 1: Try binary {-1, 1} sequences
    # Start with a random binary sequence
    np.random.seed(42)
    
    # Try several random binary sequences
    for trial in range(100):
        if time.time() > deadline:
            break
        f = np.random.choice([-1.0, 1.0], size=N)
        val = _get_max_sidelobe(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
    
    # Strategy 2: Greedy bit-flipping optimization
    if best_f is not None:
        f = best_f.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try flipping bit i
                f[i] = -f[i]
                val = _get_max_sidelobe(f)
                if val < best_val:
                    best_val = val
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] = -f[i]  # revert
    
    # Strategy 3: Try structured patterns
    patterns = []
    # Alternating
    patterns.append(np.tile([-1.0, 1.0], N // 2))
    # Blocks of 2
    patterns.append(np.tile([-1.0, -1.0, 1.0, 1.0], N // 4))
    # Blocks of 4
    patterns.append(np.tile([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], N // 8))
    # Pseudo-random based on quadratic residue
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if N % p == 0:
            seq = []
            for i in range(N):
                r = (i * i) % p
                seq.append(1.0 if r == 0 or ((r * pow(r, p-2, p)) % p == 1) else -1.0)
            patterns.append(np.array(seq))
    
    for pat in patterns:
        if len(pat) != N:
            continue
        val = _get_max_sidelobe(pat)
        if val < best_val:
            best_val = val
            best_f = pat.copy()
    
    # Fallback
    if best_f is None:
        return np.ones(N)
    
    return best_f
