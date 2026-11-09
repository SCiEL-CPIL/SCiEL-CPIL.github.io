"""Optimize squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

DEADLINE = 140.0
MAX_N = 10**18

# Generate primes up to ~2000 for the search pool
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

PRIMES = _sieve(3000)
LOG_LIMIT = math.log(MAX_N)

def _divisors_sorted(primes):
    divs = [1]
    for p in primes:
        m = len(divs)
        divs += [divs[i] * p for i in range(m)]
    divs.sort()
    return divs

def _tau_perp(divs):
    c = 0
    prev = divs[0]
    for x in divs[1:]:
        if math.gcd(prev, x) == 1:
            c += 1
        prev = x
    return c

def _product(primes):
    r = 1
    for p in primes:
        r *= p
        if r > MAX_N:
            return r
    return r

def construct():
    start = time.time()
    rng = random.Random(12345)

    best_score = -1
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

    def eval_set(primes):
        nonlocal best_score, best_n
        n = _product(primes)
        if n > MAX_N or n < 2:
            return -1
        divs = _divisors_sorted(primes)
        s = _tau_perp(divs)
        if s > best_score:
            best_score = s
            best_n = n
        return s

    # ---- Phase 1: sweep k with smallest primes to find best k ----
    for k in range(2, 17):
        if time.time() - start > DEADLINE - 60:
            break
        cand = PRIMES[:k]
        n = _product(cand)
        if n > MAX_N:
            break
        eval_set(cand)

    # ---- Phase 2: local search (1-swap) from the best known set ----
    # Determine best k from phase 1
    best_k = 2
    for k in range(2, 17):
        cand = PRIMES[:k]
        n = _product(cand)
        if n > MAX_N:
            break
        # recompute
        divs = _divisors_sorted(cand)
        s = _tau_perp(divs)
        if s > _tau_perp(_divisors_sorted(PRIMES[:best_k])):
            best_k = k

    # Local search for several k values around best_k
    for k in [max(2, best_k - 2), best_k - 1, best_k, best_k + 1, best_k + 2]:
        if time.time() - start > DEADLINE - 30:
            break
        if k < 2 or k > 16:
            continue
        # Check smallest-k feasible
        base = PRIMES[:k]
        if _product(base) > MAX_N:
            continue
        cur = list(base)
        eval_set(cur)

        # Iterate swap improvements - try ALL candidates per position, pick best
        for it in range(300):
            if time.time() - start > DEADLINE - 30:
                break
            improved = False
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = _tau_perp(_divisors_sorted(cur))
            best_swap = None
            best_swap_score = cur_score
            # For each position, find the best single swap
            for i in range(k):
                if time.time() - start > DEADLINE - 30:
                    break
                removed = cur[i]
                idx = PRIMES.index(removed) if removed in PRIMES else 0
                for j in range(max(0, idx - 5), min(len(PRIMES), idx + 80)):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        continue
                    new_list = cur[:i] + [np] + cur[i+1:]
                    new_list.sort()
                    divs = _divisors_sorted(new_list)
                    s = _tau_perp(divs)
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap = (i, np, new_list, new_n, s)
            if best_swap is not None:
                i, np, new_list, new_n, s = best_swap
                cur = new_list
                cur_n = new_n
                cur_set = set(cur)
                improved = True
                if s > best_score:
                    best_score = s
                    best_n = new_n
            if not improved:
                break

    # ---- Phase 3: random restarts / simulated annealing for a few iterations ----
    if time.time() - start < DEADLINE - 25:
        # Get current best as starting point
        # Find a k-sized set near best_n's prime factorization
        def factorize(n):
            res = []
            d = 2
            while d * d <= n:
                if n % d == 0:
                    res.append(d)
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            if n > 1:
                res.append(n)
            return res

        # Use best_n's primes as a seed
        try:
            seed_primes = factorize(best_n)
        except Exception:
            seed_primes = PRIMES[:8]

        k0 = len(seed_primes)
        for restart in range(40):
            if time.time() - start > DEADLINE - 20:
                break
            # Start from a random-ish set of size k0 or k0±1
            k = k0 + rng.choice([-1, 0, 0, 1])
            if k < 2 or k > 16:
                continue
            # Random subset of primes, biased toward small
            idx_pool = list(range(0, 60))
            rng.shuffle(idx_pool)
            chosen = sorted(PRIMES[i] for i in idx_pool[:k])
            if _product(chosen) > MAX_N:
                # Fallback to smallest k
                chosen = PRIMES[:k]
                if _product(chosen) > MAX_N:
                    continue
            cur = list(chosen)
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = eval_set(cur)

            # Hill climb with swaps, with occasional random moves (annealing-lite)
            for it in range(150):
                if time.time() - start > DEADLINE - 20:
                    break
                improved = False
                i = rng.randrange(k)
                removed = cur[i]
                idx = PRIMES.index(removed)
                # Candidate new primes - wider window
                cands = []
                for j in range(max(0, idx - 5), min(len(PRIMES), idx + 60)):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        continue
                    cands.append(np)
                if not cands:
                    continue
                np = rng.choice(cands)
                new_list = cur[:i] + [np] + cur[i+1:]
                new_list.sort()
                divs = _divisors_sorted(new_list)
                s = _tau_perp(divs)
                new_n = _product(new_list)
                if s > cur_score or (s == cur_score and rng.random() < 0.15):
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_n = new_n

    # ---- Phase 4: Final polish - exhaustive 1-swap neighborhood of best_n ----
    if time.time() - start < DEADLINE - 5:
        try:
            def factorize(n):
                res = []
                d = 2
                while d * d <= n:
                    if n % d == 0:
                        res.append(d)
                        while n % d == 0:
                            n //= d
                    d += 1 if d == 2 else 2
                if n > 1:
                    res.append(n)
                return res
            
            best_primes = factorize(best_n)
            k = len(best_primes)
            cur = sorted(best_primes)
            cur_set = set(cur)
            cur_n = best_n
            cur_score = best_score
            
            # Multiple rounds of best-swap improvement
            for round_num in range(10):
                if time.time() - start > DEADLINE - 3:
                    break
                best_swap = None
                best_swap_score = cur_score
                for i in range(k):
                    if time.time() - start > DEADLINE - 3:
                        break
                    removed = cur[i]
                    if removed not in PRIMES:
                        continue
                    idx = PRIMES.index(removed)
                    for j in range(max(0, idx - 10), min(len(PRIMES), idx + 100)):
                        np = PRIMES[j]
                        if np in cur_set:
                            continue
                        new_n = cur_n // removed * np
                        if new_n > MAX_N:
                            continue
                        new_list = cur[:i] + [np] + cur[i+1:]
                        new_list.sort()
                        divs = _divisors_sorted(new_list)
                        s = _tau_perp(divs)
                        if s > best_swap_score:
                            best_swap_score = s
                            best_swap = (i, np, new_list, new_n, s)
                if best_swap is not None:
                    i, np, new_list, new_n, s = best_swap
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    best_score = s
                    best_n = new_n
                else:
                    break
        except Exception:
            pass

    return best_n
# EVOLVE-BLOCK-END
