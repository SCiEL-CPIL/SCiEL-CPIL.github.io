"""Few prime factors across pairwise sums - set-cover + local search design."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _factorize(n):
    """Return the set of distinct prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors


def _objective(arr):
    """Count distinct primes dividing sums of distinct-index pairs."""
    s = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s |= _factorize(ai + arr[j])
    return len(s)


def _try_base(base, n=12, maximum=10000):
    """Build n distinct multiples of base <= maximum; return (arr, obj) or (None, inf)."""
    # smallest n distinct positive multiples of base
    max_mult = maximum // base
    if max_mult < n:
        return None, float('inf')
    arr = [base * k for k in range(1, n + 1)]
    return arr, _objective(arr)


def _local_search(arr, deadline, rng):
    """Local search: try replacing one element to reduce objective."""
    n = len(arr)
    maximum = 10000
    current = _objective(arr)
    best = list(arr)
    best_obj = current
    while time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() >= deadline:
                break
            old = best[idx]
            # try a few candidate replacements
            candidates = set()
            # random values
            for _ in range(40):
                candidates.add(rng.randint(1, maximum))
            # neighbors of old
            for d in (-3, -2, -1, 1, 2, 3):
                v = old + d
                if 1 <= v <= maximum:
                    candidates.add(v)
            # multiples of small bases
            for b in (2, 3, 5, 7, 10, 15, 21, 30, 42, 70, 105, 210):
                k = old // b
                for kk in (k - 1, k + 1):
                    if kk >= 1 and kk * b <= maximum:
                        candidates.add(kk * b)
            for cand in candidates:
                if cand == old:
                    continue
                if cand in best and best.index(cand) != idx:
                    continue
                trial = list(best)
                trial[idx] = cand
                obj = _objective(trial)
                if obj < best_obj:
                    best_obj = obj
                    best = trial
                    improved = True
                    break
        if not improved:
            # random perturbation to escape local minimum
            idx = rng.randrange(n)
            best[idx] = rng.randint(1, maximum)
            # ensure distinctness
            for k in range(n):
                if k != idx and best[k] == best[idx]:
                    best[idx] = best[k] + 1
                    if best[idx] > maximum:
                        best[idx] = best[k] - 1
            if best_obj > _objective(best):
                best_obj = _objective(best)
    return best, best_obj


def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90s kill
    rng = random.Random(12345)

    # Strong base: multiples of 210 (primes 2,3,5,7) -> 4 distinct primes
    base_arr, base_obj = _try_base(210)
    if base_arr is None:
        base_arr = list(range(1, 13))
        base_obj = _objective(base_arr)

    best_arr = base_arr
    best_obj = base_obj

    # Try a few more base constructions
    for b in (210, 120, 60, 30, 84, 42, 105, 70, 15, 10):
        if time.time() >= deadline:
            break
        arr, obj = _try_base(b)
        if arr is not None and obj < best_obj:
            best_arr = arr
            best_obj = obj

    # Local search to try to improve
    arr, obj = _local_search(list(best_arr), deadline, rng)
    if obj < best_obj:
        best_arr = arr
        best_obj = obj

    return best_arr

# EVOLVE-BLOCK-END
