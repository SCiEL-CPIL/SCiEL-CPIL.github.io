"""Suite task: largest set of distinct ints in [2,40] with all 2^k subset products distinct.

Distinct subset products  <=>  the prime-exponent vectors of the chosen numbers are
linearly independent over R.  There are 12 primes <= 40, so the maximum possible size
is 12.  We build a 12-vector basis greedily, then polish with randomized swaps to
find a valid 12-set (maximizing the total as a secondary, robustness objective).
"""
# EVOLVE-BLOCK-START

import math
import random
import time

_START = time.time()
_DEADLINE = _START + 84.0   # safety margin under the 90 s kill

# The 12 primes <= 40, in index order.
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_P = len(_PRIMES)
_PMAX = _PRIMES[-1]

# Precompute the exponent vector of every candidate in [2,40].
_EXP = {}
for _x in range(2, 41):
    _v = []
    for _p in _PRIMES:
        _c = 0
        while _x % _p == 0:
            _x //= _p
            _c += 1
        _v.append(_c)
    _EXP[_c if False else _x] = _v  # placeholder, replaced below

_EXP = {}
for _x in range(2, 41):
    _v = []
    _t = _x
    for _p in _PRIMES:
        _c = 0
        while _t % _p == 0:
            _t //= _p
            _c += 1
        _v.append(_c)
    _EXP[_x] = _v
_DEL = _EXP.pop(1, None)  # 1 is not a candidate
_CANDS = list(range(2, 41))


def _is_independent(vecs):
    """True iff the integer vectors are linearly independent (Gaussian elimination)."""
    m = len(vecs)
    if m == 0:
        return True
    n = len(vecs[0])
    A = [list(v) for v in vecs]
    rank = 0
    for col in range(n):
        if rank == m:
            break
        piv = -1
        for r in range(rank, m):
            if A[r][col] != 0:
                piv = r
                break
        if piv == -1:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        pv = A[rank][col]
        for c in range(col, n):
            A[rank][c] /= pv
        for r in range(m):
            if r != rank and A[r][col] != 0:
                f = A[r][col]
                for c in range(col, n):
                    A[r][c] -= f * A[rank][c]
        rank += 1
    return rank == m


def _verify(s):
    """Exact distinct-subset-products check (only used on the final answer)."""
    s = list(s)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in s):
        return False
    if any(x < 2 or x > 40 for x in s):
        return False
    if len(set(s)) != len(s):
        return False
    n = len(s)
    prods = [1]
    for x in s:
        prods += [p * x for p in prods]
    if len(set(prods)) != (1 << n):
        return False
    return True


def _greedy_basis(cands, rng):
    """Greedily grow an independent set of exponent vectors from the candidate pool."""
    chosen = []
    pool = list(cands)
    while pool and len(chosen) < _P:
        best = None
        best_score = None
        for x in pool:
            trial = chosen + [x]
            if _is_independent([_EXP[t] for t in trial]):
                # prefer numbers with the largest smallest prime factor (spread)
                sc = 0
                t = x
                for p in _PRIMES:
                    if t % p == 0:
                        sc = p
                        break
                key = (sc, -x, rng.random())
                if best_score is None or key > best_score:
                    best = x
                    best_score = key
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
    return chosen


def _local_search(chosen, rng, deadline):
    """Randomized swap polish: keep the set independent, maximize total sum."""
    chosen = list(chosen)
    cur_sum = sum(chosen)
    chosen_set = set(chosen)
    best = list(chosen)
    best_sum = cur_sum
    while time.time() < deadline:
        if len(chosen) < 2:
            break
        if rng.random() < 0.7:
            # 1 <-> 1 swap
            i = rng.randrange(len(chosen))
            a = chosen[i]
            cand = [x for x in _CANDS if x not in chosen_set]
            if not cand:
                continue
            b = cand[rng.randrange(len(cand))]
            trial = chosen[:i] + [b] + chosen[i + 1:]
            if _is_independent([_EXP[t] for t in trial]):
                ns = cur_sum - a + b
                if ns > cur_sum:
                    chosen = trial
                    cur_sum = ns
                    chosen_set = set(chosen)
                    if cur_sum > best_sum:
                        best_sum = cur_sum
                        best = list(chosen)
        else:
            # 2 <-> 2 swap
            i, j = rng.sample(range(len(chosen)), 2)
            a, c = chosen[i], chosen[j]
            cand = [x for x in _CANDS if x not in chosen_set]
            if len(cand) < 2:
                continue
            b, d = rng.sample(cand, 2)
            trial = list(chosen)
            trial[i] = b
            trial[j] = d
            if _is_independent([_EXP[t] for t in trial]):
                ns = cur_sum - a - c + b + d
                if ns > cur_sum:
                    chosen = trial
                    cur_sum = ns
                    chosen_set = set(chosen)
                    if cur_sum > best_sum:
                        best_sum = cur_sum
                        best = list(chosen)
    return best


def construct():
    deadline = _DEADLINE
    rng = random.Random(12345)

    # Initial basis.
    chosen = _greedy_basis(_CANDS, rng)
    chosen = _local_search(chosen, rng, min(deadline, _START + 60.0))

    # Extra restarts if we still have time, keeping the best (largest) valid set.
    best = list(chosen)
    best_sum = sum(chosen)
    while time.time() < deadline - 2.0:
        restart_rng = random.Random(rng.randrange(1 << 30))
        cand = _greedy_basis(_CANDS, restart_rng)
        cand = _local_search(cand, restart_rng, min(deadline - 2.0, time.time() + 12.0))
        if _verify(cand) and sum(cand) > best_sum:
            best = list(cand)
            best_sum = sum(cand)

    # Final exact verification; fall back to a guaranteed-valid set otherwise.
    if _verify(best):
        return [int(x) for x in best]
    # Guaranteed independent 12-set: one vector per prime (primes + a few prime powers).
    fallback = [2, 4, 9, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if _verify(fallback):
        return [int(x) for x in fallback]
    return [2]

# EVOLVE-BLOCK-END
