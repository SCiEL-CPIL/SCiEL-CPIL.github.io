import time
import math
import numpy as np

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _legendre_prime(n):
    """Smallest prime >= n."""
    def is_prime(m):
        if m < 2:
            return False
        if m < 4:
            return True
        if m % 2 == 0:
            return False
        if m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True
    m = max(n, 3)
    while not is_prime(m):
        m += 1
    return m


def _legendre_seq(p, N):
    """Signed step heights from Legendre symbol over a prime p ~ N.

    Nonzero residues map to +1/-1 by quadratic-residue status;
    the single zero maps to 0. This yields a balanced signed sequence
    with low additive energy, suppressing autoconvolution sidelobes.
    """
    r = np.arange(N)
    # Legendre symbol: 0 if r % p == 0, +1 if quadratic residue, -1 otherwise
    lr = np.zeros(N, dtype=np.int8)
    for i in range(N):
        a = r[i] % p
        if a == 0:
            lr[i] = 0
        else:
            lr[i] = 1 if pow(a, (p - 1) // 2, p) == 1 else -1
    return lr.astype(np.float64)


def _autoconv(s):
    """Full linear autoconvolution of s (length 2N-1)."""
    return np.convolve(s, s, mode='full')


def _score(s):
    """Upper bound on C3: max |autoconvolution| (peak, including center)."""
    ac = _autoconv(s)
    return float(np.max(np.abs(ac)))


def _local_polish(s, time_budget):
    """Flip individual bits to reduce the peak. Returns best sequence + score."""
    s = s.copy()
    best = s.copy()
    best_sc = _score(s)
    start = time.time()
    n = len(s)
    # Precompute autoconvolution
    ac = _autoconv(s)
    peak = float(np.max(np.abs(ac)))
    # Iterate: find the bin contributing to the max abs, flip the bit that
    # most reduces it. Use a greedy coordinate-descent over bit flips.
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        # current peak location
        ac = _autoconv(s)
        absac = np.abs(ac)
        pk = float(absac.max())
        if pk < best_sc:
            best_sc = pk
            best = s.copy()
        # try flipping each bit, keep best improvement
        best_j = -1
        best_after = pk
        # sample a subset of indices if too slow
        idx = np.arange(n)
        for j in idx:
            s[j] = -s[j]
            ac2 = _autoconv(s)
            v = float(np.abs(ac2).max())
            s[j] = -s[j]
            if v < best_after:
                best_after = v
                best_j = j
        if best_j >= 0 and best_after < pk - 1e-12:
            s[best_j] = -s[best_j]
            improved = True
    return best, best_sc


def construct():
    t0 = time.time()
    # Candidate primes near N to explore low additive energy
    candidates = []
    p0 = _legendre_prime(N)
    for p in (p0, p0 + 2, p0 - 2, p0 + 4, p0 - 4, p0 + 6, p0 + 8, p0 + 10):
        if p >= N and p > 3:
            candidates.append(p)
    # dedupe, keep order
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    best_seq = None
    best_sc = float('inf')
    for p in uniq:
        if time.time() > DEADLINE - 10:
            break
        s = _legendre_seq(p, N)
        sc = _score(s)
        if sc < best_sc:
            best_sc = sc
            best_seq = s.copy()
    if best_seq is None:
        best_seq = _legendre_seq(p0, N)
        best_sc = _score(best_seq)

    # Polish with remaining time budget
    remaining = DEADLINE - 8 - (time.time() - t0)
    if remaining > 1.0:
        best_seq, best_sc = _local_polish(best_seq, remaining)

    return best_seq
