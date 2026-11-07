"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NP = len(_PRIMES)
_PRIME_SET = set(_PRIMES)

def _exp_vector(n):
    v = [0] * _NP
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    return v

_VECTORS = {a: _exp_vector(a) for a in range(2, P['n'] + 1)}
_ZERO = tuple([0] * _NP)

def _distinct(nums):
    k = len(nums)
    if k == 0:
        return True
    seen = {_ZERO}
    cur = [_ZERO]
    for a in nums:
        v = _VECTORS[a]
        new = []
        for s in cur:
            t = (s[0]+v[0], s[1]+v[1], s[2]+v[2], s[3]+v[3], s[4]+v[4], s[5]+v[5],
                 s[6]+v[6], s[7]+v[7], s[8]+v[8], s[9]+v[9], s[10]+v[10], s[11]+v[11])
            if t in seen:
                return False
            seen.add(t)
            new.append(t)
        cur = new
    return True

def construct():
    deadline = time.time() + 55.0
    primes = list(_PRIMES)
    composites = sorted(a for a in range(2, P['n'] + 1) if a not in _PRIME_SET)

    best = list(primes)
    best_size = len(primes)

    def try_greedy(order):
        c = list(primes)
        for a in order:
            if time.time() > deadline:
                break
            cand = c + [a]
            if _distinct(cand):
                c = cand
        return c

    # Deterministic greedy
    g = try_greedy(composites)
    if len(g) > best_size:
        best, best_size = g, len(g)

    # Randomized greedy passes
    random.seed(1)
    for _ in range(400):
        if time.time() > deadline:
            break
        rnd = list(composites)
        random.shuffle(rnd)
        g = try_greedy(rnd)
        if len(g) > best_size:
            best, best_size = g, len(g)

    # Local search: swaps to enable more additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = list(best)
        cur_set = set(cur)
        # add
        for a in composites:
            if a in cur_set:
                continue
            cand = cur + [a]
            if _distinct(cand):
                best, best_size = cand, best_size + 1
                cur, cur_set = list(best), set(best)
                improved = True
                break
        if improved:
            continue
        # swap
        for i, x in enumerate(cur):
            for a in composites:
                if a in cur_set:
                    continue
                cand = [y for j, y in enumerate(cur) if j != i] + [a]
                if _distinct(cand):
                    best = cand
                    cur, cur_set = list(best), set(best)
                    improved = True
                    break
            if improved:
                break

    if not _distinct(best):
        return [int(a) for a in primes]
    return [int(a) for a in best]
# EVOLVE-BLOCK-END
