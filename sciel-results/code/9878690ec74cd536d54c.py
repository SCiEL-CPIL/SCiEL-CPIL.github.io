"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _autoconv_bound(f):
    """Compute max_t |f*f(t)| for a step function f on [-1/4,1/4] with N intervals."""
    h = 1.0 / 800.0  # 400 intervals on [-1/4, 1/4], total width 1/2
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    return float(np.max(np.abs(conv)))

def _candidate_m_sequence():
    """Generate an m-sequence of length N."""
    degree = 9  # 2^9 - 1 = 511 >= 400
    state = 1
    seq = []
    for _ in range(2**degree - 1):
        seq.append(1.0 if (state & 1) else -1.0)
        feedback = (state & 1) ^ ((state >> 4) & 1)
        state = (state >> 1) | (feedback << (degree - 1))
    return np.array(seq[:N])

def _candidate_legendre():
    """Legendre sequence based on quadratic residues mod p."""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    p = N
    while not is_prime(p):
        p += 1
    seq = np.zeros(N)
    for i in range(N):
        if i == 0:
            seq[i] = 1.0
        else:
            if pow(i, (p - 1) // 2, p) == 1:
                seq[i] = 1.0
            else:
                seq[i] = -1.0
    return seq

def _local_search_fast(f, max_sweeps=30):
    """Fast greedy local search: flip bits to reduce max autoconvolution."""
    f = f.copy()
    best_val = _autoconv_bound(f)
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE:
            break
        improved = False
        # Shuffle order to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > _DEADLINE:
                break
            f[i] = -f[i]
            val = _autoconv_bound(f)
            if val < best_val:
                best_val = val
                improved = True
            else:
                f[i] = -f[i]
        if not improved:
            break
    return f, best_val

def construct():
    rng = np.random.RandomState(42)
    
    # Generate diverse candidates
    candidates = [
        _candidate_m_sequence(),
        _candidate_legendre(),
        np.where(np.arange(N) % 2 == 0, 1.0, -1.0),
        np.where(np.arange(N) % 3 == 0, 1.0, -1.0),
    ]
    
    # Add random candidates
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    best = None
    best_val = float('inf')
    
    for c in candidates:
        if time.time() > _DEADLINE - 5.0:
            break
        c_opt, val = _local_search_fast(c)
        if val < best_val:
            best_val = val
            best = c_opt
    
    if best is None:
        return np.ones(N)
    return best
# EVOLVE-BLOCK-END
