import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 78.0  # safety margin under the 90s kill

_N = 40
_NUM_PRIMES = 12
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(n):
    v = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    return tuple(v)

_VECS = {}
for _n in range(2, _N + 1):
    _VECS[_n] = _factorize(_n)

_ZERO = (0,) * _NUM_PRIMES
_ALL = list(range(2, _N + 1))


def _subset_sums(vecs):
    """All subset sums of the exponent vectors (including the empty sum)."""
    sums = {0: _ZERO}
    for v in vecs:
        for s in list(sums):
            t = s + len(v)
            sums[t] = tuple(sums[s][i] + v[i] for i in range(_NUM_PRIMES))
    return list(sums.values())


def _is_valid(vecs):
    return len(set(_subset_sums(vecs))) == (1 << len(vecs))


def _greedy():
    """Greedy build: add a number whose vector preserves distinct subset sums."""
    chosen = []
    sums = {_ZERO}
    for n in _ALL:
        v = _VECS[n]
        new = set(sums)
        ok = True
        for s in sums:
            t = s + len(v)
            nv = tuple(s[i] + v[i] for i in range(_NUM_PRIMES))
            if nv in new:
                ok = False
                break
            new.add(nv)
        if ok:
            chosen.append(n)
            sums = new
    return chosen


def _local_search(start, deadline):
    best = list(start)
    best_vecs = [_VECS[n] for n in best]
    best_len = len(best)
    rng = random.Random(12345)

    while time.time() < deadline:
        # ---- insertion pass ----
        inserted = False
        for n in _ALL:
            if n in best:
                continue
            if time.time() >= deadline:
                break
            v = _VECS[n]
            sums = set(_subset_sums(best_vecs))
            ok = True
            for s in sums:
                t = s + len(v)
                nv = tuple(s[i] + v[i] for i in range(_NUM_PRIMES))
                if nv in sums:
                    ok = False
                    break
            if ok:
                best.append(n)
                best_vecs.append(v)
                best_len += 1
                inserted = True
        # ---- swap pass ----
        swapped = False
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            for j in range(len(best)):
                if i == j:
                    continue
                for n in _ALL:
                    if n in best:
                        continue
                    v = _VECS[n]
                    trial = best_vecs[:i] + [v] + best_vecs[i + 1:]
                    if _is_valid(trial):
                        best[i] = n
                        best_vecs[i] = v
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                break
        if not inserted and not swapped:
            # try a random restart with a perturbed seed
            seed = [n for n in _ALL if rng.random() < 0.4]
            g = _greedy_from(seed, deadline)
            if len(g) > best_len:
                best = g
                best_vecs = [_VECS[n] for n in best]
                best_len = len(best)
                continue
            break
    return best


def _greedy_from(seed, deadline):
    chosen = list(seed)
    sums = set(_subset_sums([_VECS[n] for n in chosen]))
    for n in _ALL:
        if n in chosen:
            continue
        v = _VECS[n]
        ok = True
        for s in sums:
            t = s + len(v)
            nv = tuple(s[i] + v[i] for i in range(_NUM_PRIMES))
            if nv in sums:
                ok = False
                break
        if ok:
            chosen.append(n)
            for s in list(sums):
                nv = tuple(s[i] + v[i] for i in range(_NUM_PRIMES))
                sums.add(nv)
    return chosen


def construct():
    deadline = _DEADLINE
    best = _greedy()
    best = _local_search(best, deadline)
    # final safety: ensure validity, else fall back to primes
    if not _is_valid([_VECS[n] for n in best]):
        best = [p for p in _PRIMES if p <= _N]
    return list(best)
# EVOLVE-BLOCK-END
