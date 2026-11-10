"""Combined solver for Erdos #689 (n=1000): doubly-covered prefix maximization.

Structural idea (inspiration 2): assign each integer x in 1..L to its two
smallest prime divisors, giving forced consistent residues to small primes
(p <= L) and "avoidance" residues to large primes (p > L) that dodge the
danger set {x, 2x}. This is a principled, non-greedy base construction.

Refinement machinery (current program): incremental coverage updates and
local search to extend the doubly-covered prefix beyond the base L.
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 115.0          # hard safety margin (evaluator kills at 150 s)
MAX_L = 100000


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)          # all 168 primes in increasing order
_BY_P = {p: i for i, p in enumerate(_PRIMES)}


def _verify(residues, limit=MAX_L):
    """Largest L with every 1..L doubly covered, capped at limit."""
    p = np.asarray(_PRIMES, dtype=np.int64)
    r = np.asarray(residues, dtype=np.int64)
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(p, r):
        rr = int(rr)
        pp = int(pp)
        if rr == 0:
            start = pp
        else:
            start = rr
            if start < 1:
                start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    ok = mark[1:limit + 1] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _compute_coverage(residues, limit):
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, residues):
        rr = int(rr)
        pp = int(pp)
        if rr == 0:
            start = pp
        else:
            start = rr
            if start < 1:
                start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    return mark


def _score_from_coverage(coverage, limit):
    ok = coverage[1:limit + 1] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _construct_two_smallest(L):
    """Base construction: assign x in 1..L to its two smallest prime divisors.
    Small primes get forced residues; large primes get avoidance residues.
    Returns (residues, ok) where ok is False if a consistency conflict occurs."""
    residues = [0] * len(_PRIMES)
    assigned = {}
    small = [p for p in _PRIMES if p <= L]

    ok = True
    for x in range(1, L + 1):
        divs = []
        m = x
        for p in small:
            if p * p > m:
                break
            if m % p == 0:
                divs.append(p)
                while m % p == 0:
                    m //= p
                if len(divs) == 2:
                    break
        if m > 1:
            divs.append(m)
        if len(divs) < 2:
            ok = False
            break
        d1, d2 = divs[0], divs[1]
        for d in (d1, d2):
            rr = x % d
            if d in assigned and assigned[d] != rr:
                ok = False
                break
            assigned[d] = rr
        if not ok:
            break
    if not ok:
        return None, False

    for d, rr in assigned.items():
        residues[_BY_P[d]] = rr

    # Large primes: choose a residue that avoids the danger set {x, 2x}.
    for idx, p in enumerate(_PRIMES):
        if p in assigned:
            continue
        if p <= L:
            # small prime not forced by the scheme; give it a benign residue
            residues[idx] = 0
            continue
        # p > L: forbid residues v in [0, p-1] that equal x mod p or 2x mod p
        # for some x in 1..L. Since L < p, x mod p = x and 2x mod p = 2x (if 2x<p)
        # or 2x - p. Build the forbidden set.
        forb = bytearray(p)
        for x in range(1, L + 1):
            forb[x] = 1
            t = 2 * x
            if t >= p:
                t -= p
            forb[t] = 1
        # pick smallest non-forbidden residue
        r = 0
        while r < p and forb[r]:
            r += 1
        residues[idx] = r
    return residues, True


def _local_search(residues, limit, budget, seed=0):
    """Local search with incremental coverage updates."""
    rng = random.Random(seed)
    n = len(_PRIMES)
    best_res = list(residues)
    coverage = _compute_coverage(best_res, limit)
    best_score = _score_from_coverage(coverage, limit)

    t_start = time.time()
    no_improve = 0
    max_no_improve = 250

    while time.time() - t_start < budget and no_improve < max_no_improve:
        i = rng.randint(0, n - 1)
        p = _PRIMES[i]
        old_r = best_res[i]

        if p > 300:
            candidates = [rng.randint(0, p - 1) for _ in range(15)]
            if old_r not in candidates:
                candidates.append(old_r)
        else:
            candidates = list(range(p))

        best_delta = 0
        best_new_r = old_r

        old_start = old_r if old_r > 0 else p
        if old_start <= limit:
            coverage[old_start::p] -= 1

        for r in candidates:
            if r == old_r:
                continue
            new_start = r if r > 0 else p
            if new_start <= limit:
                coverage[new_start::p] += 1
                s = _score_from_coverage(coverage, limit)
                coverage[new_start::p] -= 1
            else:
                s = _score_from_coverage(coverage, limit)

            delta = s - best_score
            if delta > best_delta:
                best_delta = delta
                best_new_r = r

        if old_start <= limit:
            coverage[old_start::p] += 1

        if best_delta > 0:
            new_start = best_new_r if best_new_r > 0 else p
            if new_start <= limit:
                coverage[new_start::p] += 1
            best_res[i] = best_new_r
            best_score += best_delta
            no_improve = 0
        else:
            no_improve += 1

    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()

    # Seed fallback (the known score-1.0 construction).
    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]

    # Try a range of target L for the two-smallest-divisor base construction.
    # The natural coverage length is roughly up to the largest small prime used.
    L_candidates = [300, 500, 700, 900, 1000, 1200, 1500, 2000]

    for L in L_candidates:
        if time.time() >= DEADLINE - 8:
            break

        res, ok = _construct_two_smallest(L)
        if res is None or not ok:
            continue

        limit = min(L + 2000, MAX_L)
        score = _verify(res, limit)
        if score > best_score:
            best_score = score
            best_res = list(res)

        # Refine with local search if we have time.
        if time.time() < DEADLINE - 6 and score > best_score - 50:
            lc_budget = min(3.0, max(0.5, (DEADLINE - time.time()) / 8))
            s2, r2 = _local_search(res, limit, budget=lc_budget, seed=2024 + L)
            if s2 > best_score:
                best_score = s2
                best_res = r2

        if best_score >= 1000:
            break

    # Final verification with an extended limit.
    final_score = _verify(best_res, min(best_score + 3000, MAX_L))
    if final_score > best_score:
        best_score = final_score

    return best_res
