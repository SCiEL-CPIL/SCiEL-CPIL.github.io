"""Optimize squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import random

DEADLINE = 145.0
MAX_N = 10**18

# Generate primes up to ~5000 for the search pool
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

PRIMES = _sieve(5000)
# Build a mapping from prime to index for O(1) lookup
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

def _divisors_sorted(primes):
    """Generate all divisors of a squarefree number given its prime factors, sorted."""
    divs = [1]
    for p in primes:
        m = len(divs)
        divs += [divs[i] * p for i in range(m)]
    divs.sort()
    return divs

def _tau_perp(divs):
    """Count adjacent coprime pairs in sorted divisors."""
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
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # 9699690

    def time_ok(safety=30):
        return time.time() - start < DEADLINE - safety

    def eval_set(primes, update_best=True):
        nonlocal best_score, best_n
        n = _product(primes)
        if n > MAX_N or n < 2:
            return -1
        divs = _divisors_sorted(primes)
        s = _tau_perp(divs)
        if update_best and s > best_score:
            best_score = s
            best_n = n
        return s

    # ---- Phase 1: Sweep k with smallest primes to find best k ----
    best_k = 2
    for k in range(2, 17):
        if not time_ok(60):
            break
        cand = PRIMES[:k]
        n = _product(cand)
        if n > MAX_N:
            break
        s = eval_set(cand)
        if s > best_score:
            best_k = k

    # ---- Phase 2: Exhaustive 1-swap local search ----
    # For several k values around best_k, do a thorough local search
    for k in range(max(2, best_k - 2), min(16, best_k + 3)):
        if not time_ok(40):
            break
        base = PRIMES[:k]
        if _product(base) > MAX_N:
            continue
        cur = list(base)
        cur_set = set(cur)
        cur_n = _product(cur)
        cur_score = _tau_perp(_divisors_sorted(cur))
        if cur_score > best_score:
            best_score = cur_score
            best_n = cur_n

        # Iterate swap improvements - try ALL candidates per position
        for it in range(500):
            if not time_ok(40):
                break
            improved = False
            best_swap_score = cur_score
            best_swap_new = None
            best_swap_idx = -1

            for i in range(k):
                if not time_ok(40):
                    break
                removed = cur[i]
                idx = PRIME_TO_IDX.get(removed, 0)
                # Try primes from slightly before to well after removed
                j_start = max(0, idx - 5)
                j_end = min(len(PRIMES), idx + 200)
                for j in range(j_start, j_end):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        # Since primes are sorted, if this is too big, larger ones will be too
                        # But only if np > removed; if np < removed, new_n < cur_n
                        if np > removed:
                            break
                        continue
                    new_list = cur[:i] + [np] + cur[i+1:]
                    new_list.sort()
                    divs = _divisors_sorted(new_list)
                    s = _tau_perp(divs)
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap_new = np
                        best_swap_idx = i
                        improved = True

            if not improved:
                break

            # Apply the best swap found
            if best_swap_idx >= 0:
                removed = cur[best_swap_idx]
                cur_set.remove(removed)
                cur_set.add(best_swap_new)
                cur[best_swap_idx] = best_swap_new
                cur.sort()
                cur_n = _product(cur)
                cur_score = best_swap_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_n = cur_n

    # ---- Phase 3: Simulated annealing with random restarts ----
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
        for restart in range(40):
            if not time_ok(20):
                break
            k = k0 + rng.choice([-1, 0, 0, 0, 1])
            if k < 2 or k > 16:
                continue
            # Random subset of primes, biased toward small
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
            cur_score = _tau_perp(_divisors_sorted(cur))
            if cur_score > best_score:
                best_score = cur_score
                best_n = cur_n

            # Simulated annealing
            temp = 3.0
            for it in range(200):
                if not time_ok(20):
                    break
                i = rng.randrange(k)
                removed = cur[i]
                idx = PRIME_TO_IDX.get(removed, 0)
                # Candidate new primes - wider neighborhood
                cands = []
                for j in range(max(0, idx - 10), min(len(PRIMES), idx + 100)):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        if np > removed:
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
                    if delta < 15 and rng.random() < math.exp(-delta / temp):
                        cur = new_list
                        cur_n = new_n
                        cur_set = set(cur)
                        cur_score = s
                temp *= 0.97

    # ---- Phase 4: Final polish - exhaustive 1-swap from best_n ----
    if time_ok(10):
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
            cur = factorize(best_n)
        except Exception:
            cur = PRIMES[:8]

        k = len(cur)
        cur_set = set(cur)
        cur_n = _product(cur)
        cur_score = _tau_perp(_divisors_sorted(cur))

        # Exhaustive 1-swap improvement with wide window
        for it in range(1000):
            if not time_ok(10):
                break
            improved = False
            best_swap_score = cur_score
            best_swap_new = None
            best_swap_idx = -1

            for i in range(k):
                if not time_ok(10):
                    break
                removed = cur[i]
                idx = PRIME_TO_IDX.get(removed, 0)
                j_start = max(0, idx - 20)
                j_end = min(len(PRIMES), idx + 300)
                for j in range(j_start, j_end):
                    np = PRIMES[j]
                    if np in cur_set:
                        continue
                    new_n = cur_n // removed * np
                    if new_n > MAX_N:
                        if np > removed:
                            break
                        continue
                    new_list = cur[:i] + [np] + cur[i+1:]
                    new_list.sort()
                    divs = _divisors_sorted(new_list)
                    s = _tau_perp(divs)
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap_new = np
                        best_swap_idx = i
                        improved = True

            if not improved:
                break

            if best_swap_idx >= 0:
                removed = cur[best_swap_idx]
                cur_set.remove(removed)
                cur_set.add(best_swap_new)
                cur[best_swap_idx] = best_swap_new
                cur.sort()
                cur_n = _product(cur)
                cur_score = best_swap_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_n = cur_n

    return best_n
# EVOLVE-BLOCK-END
