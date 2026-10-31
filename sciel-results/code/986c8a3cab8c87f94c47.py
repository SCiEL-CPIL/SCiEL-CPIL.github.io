"""Rewritten program: Barker/Legendre seeds + smooth surrogate + flip local search."""
import numpy as np
import time

N = 400

_START = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

# ---------- objective (exact FFT autoconvolution bound) ----------
def _autoconv_bound(f):
    """max_t |f*f(t)| for a step function f (length N) on [-1/4,1/4], interval h=1/200."""
    h = 1.0 / 200.0
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    return float(np.max(np.abs(conv)))


def _smooth_bound(f, beta=8.0):
    """Soft-max surrogate of max_t |f*f(t)|, smooth in f, for gradient guidance."""
    h = 1.0 / 200.0
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    a = np.abs(conv)
    m = a.max()
    # log-sum-exp scaling: beta * (m + log(sum exp(beta*(a-m)))) / beta
    return float(m + np.log(np.sum(np.exp(beta * (a - m)))) / beta)


# ---------- candidate generators ----------
def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _candidate_quadratic_residue():
    p = N
    while not _is_prime(p):
        p += 1
    res = set((x * x) % p for x in range(1, p))
    seq = np.array([1.0 if (i + 1) in res else -1.0 for i in range(N)])
    return seq


def _candidate_legendre():
    # Legendre sequence: +1 if i is quadratic residue mod p, -1 otherwise, for i in 1..N
    p = N
    while not _is_prime(p):
        p += 1
    seq = np.ones(N)
    for i in range(N):
        x = i + 1
        if pow(x, (p - 1) // 2, p) == p - 1:
            seq[i] = -1.0
    return seq


def _candidate_barker_like():
    # Barker sequences (length 13) tiled / repeated to length N with sign symmetry
    barker = np.array([1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1])
    seq = np.tile(barker, N // len(barker) + 1)[:N]
    return seq.astype(float)


def _candidate_random(seed):
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)


def _candidate_alternating():
    return np.where(np.arange(N) % 2 == 0, 1.0, -1.0)


# ---------- flip local search ----------
def _local_search(f, max_rounds=40, use_smooth=False, beta=8.0):
    """Coordinate-descent: flip a random subset of positions each round, keep if bound improves."""
    f = f.copy()
    cur = _smooth_bound(f, beta) if use_smooth else _autoconv_bound(f)
    rng = np.random.RandomState(12345)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < _DEADLINE:
        improved = False
        rounds += 1
        # try a random permutation of positions, flip and accept greedily
        idx = rng.permutation(N)
        for k in idx:
            if time.time() > _DEADLINE:
                break
            f[k] = -f[k]
            new = _smooth_bound(f, beta) if use_smooth else _autoconv_bound(f)
            if new < cur - 1e-12:
                cur = new
                improved = True
            else:
                f[k] = -f[k]  # revert
    return f, cur


def _local_search_fast(f, max_rounds=30):
    """Faster: evaluate full bound only after a batch of accepted flips (reduces FFT calls)."""
    f = f.copy()
    cur = _autoconv_bound(f)
    rng = np.random.RandomState(777)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < _DEADLINE:
        improved = False
        rounds += 1
        idx = rng.permutation(N)
        pending = []
        for k in idx:
            if time.time() > _DEADLINE:
                break
            f[k] = -f[k]
            pending.append(k)
            # recompute every 8 flips
            if len(pending) >= 8:
                new = _autoconv_bound(f)
                if new < cur - 1e-12:
                    cur = new
                    improved = True
                    pending = []
                else:
                    for kk in pending:
                        f[kk] = -f[kk]
                    pending = []
    return f, cur


def construct():
    candidates = []
    candidates.append(_candidate_quadratic_residue())
    candidates.append(_candidate_legendre())
    candidates.append(_candidate_barker_like())
    candidates.append(_candidate_alternating())
    for s in range(12):
        candidates.append(_candidate_random(2000 + s))

    best = None
    best_val = float('inf')

    for c in candidates:
        if time.time() > _DEADLINE:
            break
        # polish with fast local search
        try:
            f, v = _local_search_fast(c, max_rounds=25)
        except Exception:
            f, v = c, _autoconv_bound(c)
        if v < best_val:
            best_val = v
            best = f

    # final high-precision polish on the best with the exact objective
    if best is not None and time.time() < _DEADLINE:
        try:
            f2, v2 = _local_search(best, max_rounds=30, use_smooth=False)
            if v2 < best_val:
                best, best_val = f2, v2
        except Exception:
            pass

    if best is None:
        return np.ones(N)
    return best
