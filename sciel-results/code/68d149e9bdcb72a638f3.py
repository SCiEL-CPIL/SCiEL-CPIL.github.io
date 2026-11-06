"""Distinct subset sums, n=12, bounded max.
Search for a small-M 'complementary' complete set S: 2S is a Sidon set in [0,2M].
Verified directly by enumerating all 4096 subset sums of the shifted set.
"""
import time
import random
from itertools import combinations

N = 12
NSUB = 1 << N
MAXV = 1048576
DEADLINE = 82.0          # hard stop well under the 90 s kill
DEADLINE_SOFT = 70.0     # stop searching once a good solution is in hand
_seed_state = None

# ---- fast bit-difference enumeration (all 4096 subset sums of a list) ----
def _subset_sums(arr):
    n = len(arr)
    sums = [0] * (1 << n)
    for i in range(n):
        b = arr[i]
        h = 1 << i
        for j in range(h):
            sums[h | j] = sums[j] + b
    return sums


def _valid_shifted(vals):
    if len(vals) != N or any(v <= 0 for v in vals):
        return False
    if len(set(_subset_sums(vals))) != NSUB:
        return False
    return max(vals) <= MAXV


def _is_complete(S, M):
    """All 4096 subset sums of S distinct and within [0, 2M] (=> 2S Sidon)."""
    ss = _subset_sums(S)
    return len(set(ss)) == NSUB and max(ss) <= 2 * M


def _sidon_complete(M):
    """Randomly search for a size-12 complete set S in [0, M] (2S Sidon)."""
    global _seed_state
    if _seed_state is None:
        _seed_state = random.Random(12345)
    rng = _seed_state
    base = list(range(M + 1))
    attempts = 0
    while True:
        S = sorted(rng.sample(base, N))
        attempts += 1
        if _is_complete(S, M):
            return S
        if time.time() >= DEADLINE:
            return None
        if attempts % 200000 == 0:
            rng.seed()  # re-randomize to escape plateaus


def _greedy_complete(M):
    """Greedy build of a size-12 complete set in [0, M]."""
    S = []
    for _ in range(N):
        for v in range(M + 1):
            T = S + [v]
            if _is_complete(T, M):
                S = T
                break
        else:
            return None
        if time.time() >= DEADLINE:
            return None
    return S


def _verify_and_shift(S, M):
    """Shift S to positive ints and confirm all 4096 sums distinct + bounded."""
    # shift so min becomes 1
    off = 1 - min(S)
    vals = sorted(s + off for s in S)
    if max(vals) > MAXV:
        return None
    if _valid_shifted(vals):
        return vals
    return None


def _try_m(M):
    if time.time() >= DEADLINE:
        return None
    S = _sidon_complete(M)
    if S is None:
        S = _greedy_complete(M)
    if S is None:
        return None
    return _verify_and_shift(S, M)


def construct():
    global _seed_state
    t0 = time.time()
    _seed_state = random.Random(987654321)

    # Known-good fallback: powers of two (max 2048), always valid.
    best = [1 << i for i in range(N)]

    # Try increasingly large M; smaller M gives smaller max element.
    for M in range(4200, 7201, 5):
        if time.time() >= DEADLINE:
            break
        vals = _try_m(M)
        if vals is not None:
            if max(vals) < max(best):
                best = vals
                if max(vals) < 1200:  # strong result; stop early
                    break
        if time.time() >= DEADLINE_SOFT:
            break

    if time.time() >= DEADLINE:
        return best
    # safety: ensure the returned set is valid
    if not _valid_shifted(best):
        best = [1 << i for i in range(N)]
    return best
