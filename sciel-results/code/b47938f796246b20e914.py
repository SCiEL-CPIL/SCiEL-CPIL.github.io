"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _autoconv_bound(f):
    """Compute max_t |f*f(t)| for a step function f on [-1/4,1/4] with N intervals.
    f is an array of N heights. Interval width h = 1/200.
    The continuous autoconvolution (f*f)(t) = int f(s) f(t-s) ds.
    We evaluate on a fine grid and return the max absolute value.
    """
    h = 1.0 / 200.0
    # f is defined on [-1/4, 1/4]; sample centers
    # Use FFT-based convolution for efficiency.
    # Pad to length >= 2N for linear convolution.
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h  # scale by h for the integral
    # The convolution is defined for t in [-1/4, 1/4] + [-1/4,1/4] = [-1/2, 1/2]
    # conv[k] corresponds to t = (k - N + 1) * h approximately; we take max abs over the valid range
    return float(np.max(np.abs(conv)))

def _candidate_quadratic_residue():
    """±1 sequence based on quadratic residues mod a prime near N."""
    # Find a prime p >= N
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
    # Quadratic residues mod p
    res = set()
    for x in range(1, p):
        res.add((x * x) % p)
    # Build sequence of length N
    seq = np.zeros(N)
    for i in range(N):
        seq[i] = 1.0 if (i + 1) in res else -1.0
    return seq

def _candidate_random(seed):
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _candidate_alternating():
    return np.where(np.arange(N) % 2 == 0, 1.0, -1.0)

def construct():
    candidates = []
    candidates.append(_candidate_quadratic_residue())
    candidates.append(_candidate_alternating())
    # Add a few random candidates
    for s in range(20):
        candidates.append(_candidate_random(1000 + s))
    
    best = None
    best_val = float('inf')
    for c in candidates:
        if time.time() > _DEADLINE:
            break
        v = _autoconv_bound(c)
        if v < best_val:
            best_val = v
            best = c
    if best is None:
        # Fallback: all ones (seed behavior)
        return np.ones(N)
    return best
# EVOLVE-BLOCK-END
