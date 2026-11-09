import time
import random
import math
from math import gcd
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 145.0
_N = 512
_LIMIT = 511


def _prime_sieve(n):
    """Return set of primes up to n-1 (i.e., primes < n)."""
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n, i):
                is_prime[j] = False
    return [i for i in range(2, n) if is_prime[i]]


def _factorize(n, primes):
    """Return set of prime factors of n. If n > 1 after division, it's a prime factor."""
    factors = set()
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return factors


def _precompute():
    """Precompute prime factors for all numbers 1..511."""
    primes = _prime_sieve(_N)
    factor_cache = {}
    for a in range(1, _LIMIT + 1):
        factor_cache[a] = _factorize(a, primes)
    # Precompute weights
    weights = {a: 1.0 / (_N - a) for a in range(1, _LIMIT + 1)}
    # Order by descending weight
    order = list(range(_LIMIT, 0, -1))
    return factor_cache, weights, order


def _greedy_bitmask(factor_cache, weights, order):
    """Greedy: pick numbers in descending weight order, keeping only coprime ones."""
    chosen = []
    chosen_factors = set()
    for a in order:
        fa = factor_cache[a]
        if not (fa & chosen_factors):
            chosen.append(a)
            chosen_factors |= fa
    return chosen


def _local_search(factor_cache, weights, start, time_budget):
    """Local search with bitmask-based coprimality. Try to add elements and do swaps."""
    cur = list(start)
    cur_factors = set()
    for a in cur:
        cur_factors |= factor_cache[a]
    cur_score = sum(weights[a] for a in cur)
    end = time.time() + time_budget
    best = (cur_score, list(cur))
    best_factors = set(cur_factors)

    while time.time() < end:
        improved = False

        # Try to add elements (sorted by descending weight)
        used = set(cur)
        candidates = [a for a in order if a not in used]
        for a in candidates:
            if time.time() > end:
                break
            fa = factor_cache[a]
            if not (fa & cur_factors):
                cur.append(a)
                cur_factors |= fa
                cur_score += weights[a]
                improved = True
                if cur_score > best[0]:
                    best = (cur_score, list(cur))
                    best_factors = set(cur_factors)

        # Try 1-swap: remove one element, add a better one
        if time.time() > end:
            break
        for i in range(len(cur)):
            if time.time() > end:
                break
            a_rem = cur[i]
            fa_rem = factor_cache[a_rem]
            # Temporarily remove a_rem
            temp_factors = set(cur_factors - fa_rem)
            # Rebuild: need to recompute from scratch since factors may overlap
            # Actually, factors can overlap, so we need to rebuild
            temp_factors = set()
            for j, a in enumerate(cur):
                if j != i:
                    temp_factors |= factor_cache[a]

            best_add = None
            best_add_w = weights[a_rem]  # must be strictly better
            for a_add in order:
                if a_add == a_rem or a_add in cur:
                    continue
                fa_add = factor_cache[a_add]
                if not (fa_add & temp_factors):
                    if weights[a_add] > best_add_w:
                        best_add = a_add
                        best_add_w = weights[a_add]
                        break  # first better one in descending order

            if best_add is not None:
                cur[i] = best_add
                cur_score = cur_score - weights[a_rem] + weights[best_add]
                cur_factors = set()
                for a in cur:
                    cur_factors |= factor_cache[a]
                improved = True
                if cur_score > best[0]:
                    best = (cur_score, list(cur))
                    best_factors = set(cur_factors)

        # Try 2-swap: remove two elements, add one or two better ones
        if time.time() > end:
            break
        if len(cur) >= 2:
            # Try removing pairs with low combined weight
            cur_sorted = sorted(range(len(cur)), key=lambda i: weights[cur[i]])
            tried_pairs = 0
            for i_idx in range(len(cur_sorted)):
                if time.time() > end or tried_pairs >= 10:
                    break
                for j_idx in range(i_idx + 1, len(cur_sorted)):
                    if time.time() > end or tried_pairs >= 10:
                        break
                    tried_pairs += 1
                    a1 = cur[cur_sorted[i_idx]]
                    a2 = cur[cur_sorted[j_idx]]
                    # Remove both
                    temp_factors = set()
                    for a in cur:
                        if a != a1 and a != a2:
                            temp_factors |= factor_cache[a]
                    rem_score = weights[a1] + weights[a2]

                    # Try adding one element
                    best_add1 = None
                    best_add1_w = 0.0
                    for a_add in order:
                        if a_add == a1 or a_add == a2 or a_add in cur:
                            continue
                        fa_add = factor_cache[a_add]
                        if not (fa_add & temp_factors):
                            if weights[a_add] > rem_score:
                                best_add1 = a_add
                                best_add1_w = weights[a_add]
                                break

                    if best_add1 is not None:
                        cur = [a for a in cur if a != a1 and a != a2]
                        cur.append(best_add1)
                        cur_score = cur_score - rem_score + best_add1_w
                        cur_factors = set()
                        for a in cur:
                            cur_factors |= factor_cache[a]
                        improved = True
                        if cur_score > best[0]:
                            best = (cur_score, list(cur))
                            best_factors = set(cur_factors)
                        break

                    # Try adding two elements
                    best_pair = None
                    best_pair_w = 0.0
                    for a_add1 in order:
                        if time.time() > end:
                            break
                        if a_add1 == a1 or a_add1 == a2 or a_add1 in cur:
                            continue
                        fa1 = factor_cache[a_add1]
                        if fa1 & temp_factors:
                            continue
                        for a_add2 in order:
                            if a_add2 == a_add1 or a_add2 == a1 or a_add2 == a2 or a_add2 in cur:
                                continue
                            fa2 = factor_cache[a_add2]
                            if fa2 & temp_factors or (fa1 & fa2):
                                continue
                            pair_w = weights[a_add1] + weights[a_add2]
                            if pair_w > rem_score:
                                best_pair = (a_add1, a_add2)
                                best_pair_w = pair_w
                                break
                        if best_pair:
                            break

                    if best_pair:
                        cur = [a for a in cur if a != a1 and a != a2]
                        cur.extend(best_pair)
                        cur_score = cur_score - rem_score + best_pair_w
                        cur_factors = set()
                        for a in cur:
                            cur_factors |= factor_cache[a]
                        improved = True
                        if cur_score > best[0]:
                            best = (cur_score, list(cur))
                            best_factors = set(cur_factors)
                        break

        # Perturbation: remove a low-weight element
        if not improved and len(cur) > 5 and time.time() < end:
            cur.sort(key=lambda a: weights[a])
            cur.pop(0)
            cur_factors = set()
            for a in cur:
                cur_factors |= factor_cache[a]
            cur_score = sum(weights[a] for a in cur)

    return best[1], best[0]


def _random_restart(factor_cache, weights, order, rng):
    """Weighted random selection with coprimality constraint."""
    chosen = []
    chosen_factors = set()
    pool = list(range(1, _LIMIT + 1))
    while pool and time.time() < _DEADLINE:
        # Weight by 1/(512-a), bias toward large a
        valid = []
        for a in pool:
            fa = factor_cache[a]
            if not (fa & chosen_factors):
                valid.append(a)
        if not valid:
            break
        # Weighted random pick
        ws = [weights[a] for a in valid]
        tw = sum(ws)
        r = rng.random() * tw
        acc = 0.0
        idx = 0
        for k, a in enumerate(valid):
            acc += ws[k]
            if r <= acc:
                idx = k
                break
        a = valid[idx]
        pool.remove(a)
        chosen.append(a)
        chosen_factors |= factor_cache[a]
    return chosen


def construct():
    factor_cache, weights, order = _precompute()

    # Greedy initialization
    best = _greedy_bitmask(factor_cache, weights, order)
    best_score = sum(weights[a] for a in best)

    rng = random.Random(42)

    # Multiple random restarts with local search
    for _ in range(30):
        if time.time() > _DEADLINE - 15:
            break
        cand = _random_restart(factor_cache, weights, order, rng)
        cand, s = _local_search(factor_cache, weights, cand, time_budget=3)
        if s > best_score + 1e-12:
            best, best_score = cand, s

    # Final local search on the best
    if time.time() < _DEADLINE - 5:
        best, best_score = _local_search(factor_cache, weights, best, time_budget=10)

    return list(best)

# EVOLVE-BLOCK-END
