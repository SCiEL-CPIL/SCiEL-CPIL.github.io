"""Optimize squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

DEADLINE = 143.0
MAX_N = 10**18

# Generate primes up to 5000 for broader search pool
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

PRIMES = _sieve(5000)
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES)}

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

    def time_ok(safety=30):
        return time.time() - start < DEADLINE - safety

    # ---- Phase 1: sweep k with smallest primes to find best k ----
    for k in range(2, 17):
        if not time_ok(60):
            break
        cand = PRIMES[:k]
        n = _product(cand)
        if n > MAX_N:
            break
        eval_set(cand)

    # ---- Phase 2: local search (1-swap) from the best known set ----
    best_k = 2
    best_k_score = -1
    for k in range(2, 17):
        cand = PRIMES[:k]
        n = _product(cand)
        if n > MAX_N:
            break
        divs = _divisors_sorted(cand)
        s = _tau_perp(divs)
        if s > best_k_score:
            best_k_score = s
            best_k = k

    # Local search for several k values around best_k
    for k in [max(2, best_k - 2), best_k - 1, best_k, best_k + 1, best_k + 2]:
        if not time_ok(40):
            break
        if k < 2 or k > 16:
            continue
        base = PRIMES[:k]
        if _product(base) > MAX_N:
            continue
        cur = list(base)
        eval_set(cur)
        cur_score = _tau_perp(_divisors_sorted(cur))

        # Iterate swap improvements with expanded neighborhood
        for it in range(300):
            if not time_ok(40):
                break
            improved = False
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = _tau_perp(_divisors_sorted(cur))
            for i in range(k):
                if not time_ok(40):
                    break
                removed = cur[i]
                idx = PRIME_INDEX.get(removed, 0)
                for j in range(idx + 1, min(len(PRIMES), idx + 100)):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        break
                    if np > removed * 20:
                        break
                    new_list = cur[:i] + [np] + cur[i+1:]
                    new_list.sort()
                    divs = _divisors_sorted(new_list)
                    s = _tau_perp(divs)
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                    if s > cur_score:
                        cur = new_list
                        cur_n = new_n
                        cur_set = set(cur)
                        cur_score = s
                        improved = True
                        break
            if not improved:
                break

    # ---- Phase 3: simulated annealing with wider neighborhood ----
    if time_ok(25):
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

        try:
            seed_primes = factorize(best_n)
        except Exception:
            seed_primes = PRIMES[:8]

        k0 = len(seed_primes)
        for restart in range(50):
            if not time_ok(20):
                break
            k = k0 + rng.choice([-1, 0, 0, 1])
            if k < 2 or k > 16:
                continue
            idx_pool = list(range(0, min(len(PRIMES), 150)))
            rng.shuffle(idx_pool)
            chosen = sorted(PRIMES[i] for i in idx_pool[:k])
            if _product(chosen) > MAX_N:
                chosen = PRIMES[:k]
                if _product(chosen) > MAX_N:
                    continue
            cur = list(chosen)
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_score = eval_set(cur)

            temp = 3.0
            for it in range(150):
                if not time_ok(20):
                    break
                i = rng.randrange(k)
                removed = cur[i]
                idx = PRIME_INDEX.get(removed, 0)
                cands = []
                for j in range(max(0, idx - 8), min(len(PRIMES), idx + 80)):
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
                if s > cur_score:
                    cur = new_list
                    cur_n = new_n
                    cur_set = set(cur)
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                else:
                    delta = cur_score - s
                    if delta < 10 and rng.random() < math.exp(-delta / temp):
                        cur = new_list
                        cur_n = new_n
                        cur_set = set(cur)
                        cur_score = s
                temp *= 0.97

    # ---- Phase 4: final polish with 2-swap moves on best_n ----
    if time_ok(10):
        def factorize2(n):
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

        try:
            cur = sorted(factorize2(best_n))
        except Exception:
            cur = list(PRIMES[:8])

        cur_set = set(cur)
        cur_n = _product(cur)
        cur_score = _tau_perp(_divisors_sorted(cur))

        for it in range(200):
            if not time_ok(8):
                break
            k = len(cur)
            improved = False
            for i in range(k):
                if not time_ok(8):
                    break
                for j in range(i + 1, k):
                    if not time_ok(8):
                        break
                    ri, rj = cur[i], cur[j]
                    ii = PRIME_INDEX.get(ri, 0)
                    ij = PRIME_INDEX.get(rj, 0)
                    for a in range(max(0, ii - 5), min(len(PRIMES), ii + 40)):
                        if not time_ok(8):
                            break
                        na = PRIMES[a]
                        if na in cur_set:
                            continue
                        for b in range(max(0, ij - 5), min(len(PRIMES), ij + 40)):
                            nb = PRIMES[b]
                            if nb in cur_set or nb == na:
                                continue
                            new_n = cur_n // ri // rj * na * nb
                            if new_n > MAX_N:
                                continue
                            new_list = sorted([p for p in cur if p != ri and p != rj] + [na, nb])
                            divs = _divisors_sorted(new_list)
                            s = _tau_perp(divs)
                            if s > best_score:
                                best_score = s
                                best_n = new_n
                            if s > cur_score:
                                cur = new_list
                                cur_set = set(cur)
                                cur_n = new_n
                                cur_score = s
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    return best_n
# EVOLVE-BLOCK-END
