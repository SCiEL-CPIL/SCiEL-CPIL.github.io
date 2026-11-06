import math
import random
import time
from itertools import combinations

N = 12
MAXV = 10000
BASE_PRIMES = (2, 3, 5, 7, 11)
_BASE_SQ = [p * p for p in BASE_PRIMES]


def _is_smooth(x):
    """True if x is divisible only by primes in BASE_PRIMES (x>=1)."""
    for p in BASE_PRIMES:
        while x % p == 0:
            x //= p
    return x == 1


def _is_smooth_cache(x, cache):
    v = cache.get(x)
    if v is None:
        v = _is_smooth(x)
        cache[x] = v
    return v


def _eval_set(arr, cache):
    """Return number of distinct base primes used across all pairwise sums."""
    used = 0
    m = len(arr)
    for i in range(m):
        ai = arr[i]
        for j in range(i + 1, m):
            s = ai + arr[j]
            if not _is_smooth_cache(s, cache):
                return None
            for k, p in enumerate(BASE_PRIMES):
                if s % p == 0:
                    used |= (1 << k)
    return bin(used).count("1")


def _count_good(arr, cache):
    """Number of pairwise sums that are p-smooth."""
    m = len(arr)
    c = 0
    for i in range(m):
        ai = arr[i]
        for j in range(i + 1, m):
            if _is_smooth_cache(ai + arr[j], cache):
                c += 1
    return c


def _candidate_pool():
    """Candidate numbers whose pairwise sums with themselves are even (i.e. odd numbers)."""
    pool = []
    for x in range(1, MAXV + 1, 2):
        if _is_smooth(x + x):  # 2x must be smooth
            pool.append(x)
    return pool


def _local_search(start, pool, cache, deadline, rng):
    arr = list(start)
    arr.sort()
    cur = _eval_set(arr, cache)
    best = list(arr)
    best_score = cur
    if best_score == 1:
        return best, best_score
    while time.time() < deadline:
        improved = False
        # random order of positions
        order = list(range(len(arr)))
        rng.shuffle(order)
        for pos in order:
            old = arr[pos]
            # try replacing with a random candidate
            for _ in range(40):
                cand = pool[rng.randrange(len(pool))]
                if cand == old:
                    continue
                if any(cand == v for v in arr):
                    continue
                arr[pos] = cand
                s = _eval_set(arr, cache)
                if s is not None and (best_score is None or s < best_score):
                    best = list(arr)
                    best_score = s
                    improved = True
                    if best_score == 1:
                        arr[pos] = old
                        return best, best_score
                    break
                # also accept if it keeps smoothness count high (soft surrogate)
                arr[pos] = old
        if not improved:
            # restart from best with perturbation
            arr = list(best)
            for _ in range(3):
                p = rng.randrange(len(arr))
                cand = pool[rng.randrange(len(pool))]
                if cand not in arr:
                    arr[p] = cand
    return best, best_score


def _greedy_build(pool, cache, deadline, rng):
    """Build a set greedily, preferring numbers that keep pairwise sums smooth."""
    arr = []
    # start from a random candidate
    arr.append(pool[rng.randrange(len(pool))])
    while len(arr) < N and time.time() < deadline:
        # score each candidate by how many smooth pairwise sums it adds
        best_cand = None
        best_gain = -1
        # sample a subset to keep it fast
        sample = rng.sample(pool, min(200, len(pool)))
        for cand in sample:
            if cand in arr:
                continue
            gain = 0
            ok = True
            for v in arr:
                s = cand + v
                if _is_smooth_cache(s, cache):
                    gain += 1
                else:
                    # penalize but don't hard reject (soft)
                    gain -= 2
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand is None:
            break
        arr.append(best_cand)
    return arr


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s
    rng = random.Random(12345)
    cache = {}

    pool = _candidate_pool()
    if len(pool) < N:
        # fallback: all odd numbers
        pool = list(range(1, MAXV + 1, 2))

    best_arr = None
    best_score = None

    # Phase 1: greedy builds from multiple random starts
    for _ in range(60):
        if time.time() > deadline:
            break
        arr = _greedy_build(pool, cache, deadline, rng)
        if len(arr) < N:
            continue
        s = _eval_set(arr, cache)
        if s is not None and (best_score is None or s < best_score):
            best_score = s
            best_arr = list(arr)
            if best_score == 1:
                break

    # Phase 2: local search from best and from additional starts
    starts = []
    if best_arr is not None:
        starts.append(best_arr)
    for _ in range(15):
        if time.time() > deadline:
            break
        arr = _greedy_build(pool, cache, deadline, rng)
        if len(arr) == N:
            starts.append(arr)

    for st in starts:
        if time.time() > deadline:
            break
        arr, sc = _local_search(st, pool, cache, deadline, rng)
        if sc is not None and (best_score is None or sc < best_score):
            best_score = sc
            best_arr = list(arr)
            if best_score == 1:
                break

    # Phase 3: if no valid set found, do a more aggressive random local search
    if best_arr is None:
        # build a valid set by ensuring all sums are smooth
        # fallback: use a known structured approach
        # Try: numbers of form 2^a * 3^b * 5^c that are odd (so 2 not in them)
        # Actually odd smooth numbers: 3^a * 5^b * 7^c * 11^d
        odd_smooth = []
        for a in range(0, 10):
            v3 = 3 ** a
            for b in range(0, 8):
                v5 = v3 * (5 ** b)
                for c in range(0, 6):
                    v7 = v5 * (7 ** c)
                    for d in range(0, 4):
                        v11 = v7 * (11 ** d)
                        if v11 <= MAXV:
                            odd_smooth.append(v11)
        odd_smooth = sorted(set(odd_smooth))
        if len(odd_smooth) >= N:
            # pick N distinct, check
            for trial in range(500):
                if time.time() > deadline:
                    break
                arr = rng.sample(odd_smooth, N)
                s = _eval_set(arr, cache)
                if s is not None and (best_score is None or s < best_score):
                    best_score = s
                    best_arr = list(arr)
                    if best_score == 1:
                        break

    # Final fallback: return something valid (even if not optimal)
    if best_arr is None or len(best_arr) != N:
        # return first N odd numbers
        best_arr = list(range(1, 2 * N + 1, 2))

    # Ensure distinct and within bounds
    best_arr = sorted(set(best_arr))
    while len(best_arr) < N:
        # add a new odd number
        for x in range(1, MAXV + 1, 2):
            if x not in best_arr:
                best_arr.append(x)
                break
        best_arr = sorted(best_arr)
    best_arr = best_arr[:N]

    return best_arr
