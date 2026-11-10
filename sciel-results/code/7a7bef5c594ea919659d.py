"""Bootstrap DIRECT solver for Erdos #689 (n=1000) doubly-covered run.

Improved construction:
  * Uses a smarter greedy + local search approach to maximize doubly-covered prefix.
  * Faster verification using vectorized NumPy.
  * Handles primes (numbers with one prime factor) by using free large primes.
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 118.0          # hard safety margin (evaluator kills at 150 s)
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
    """Fast count of how many x in 1..limit satisfy >= 2 congruences.

    Returns the largest L with every 1..L doubly covered, capped at limit.
    """
    p = np.asarray(_PRIMES, dtype=np.int64)
    r = np.asarray(residues, dtype=np.int64)
    # For each x, the number of primes that hit x.
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(p, r):
        rr = int(rr)
        # Find first x in [1,limit] with x % pp == rr
        if rr == 0:
            start = pp
        else:
            start = rr
            if start < 1:
                start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    # largest L with min(mark[1..L]) >= 2
    ok = mark[1:] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _compute_coverage(residues, limit):
    """Compute coverage array for 1..limit."""
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, residues):
        rr = int(rr)
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
    """Get the doubly-covered prefix length from a coverage array."""
    ok = coverage[1:limit + 1] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _initial_greedy(target_L, seed=0):
    """Greedy construction: assign residues to maximize coverage of 1..target_L."""
    rng = random.Random(seed)
    n = len(_PRIMES)
    residues = [0] * n
    
    limit = min(target_L + 100, 5000)
    coverage = _compute_coverage(residues, limit)
    
    for _pass in range(3):
        order = list(range(n))
        if _pass == 0:
            order.sort(key=lambda i: _PRIMES[i])
        elif _pass == 1:
            order.sort(key=lambda i: -_PRIMES[i])
        else:
            rng.shuffle(order)
        
        for i in order:
            p = _PRIMES[i]
            old_r = residues[i]
            
            old_start = old_r if old_r > 0 else p
            if old_start <= limit:
                coverage[old_start::p] -= 1
            
            best_r = old_r
            best_score = -1
            
            if p > 200:
                candidates = list(range(p))
                rng.shuffle(candidates)
                candidates = candidates[:min(50, p)]
            else:
                candidates = list(range(p))
            
            for r in candidates:
                new_start = r if r > 0 else p
                if new_start <= limit:
                    coverage[new_start::p] += 1
                    s = int(np.sum(coverage[1:limit + 1] >= 2))
                    coverage[new_start::p] -= 1
                else:
                    s = int(np.sum(coverage[1:limit + 1] >= 2))
                if s > best_score:
                    best_score = s
                    best_r = r
            
            residues[i] = best_r
            new_start = best_r if best_r > 0 else p
            if new_start <= limit:
                coverage[new_start::p] += 1
    
    return residues


def _local_search(residues, limit, budget, seed=0):
    """Local search with incremental coverage updates."""
    rng = random.Random(seed)
    n = len(_PRIMES)
    best_res = list(residues)
    coverage = _compute_coverage(best_res, limit)
    best_score = _score_from_coverage(coverage, limit)
    
    t_start = time.time()
    no_improve = 0
    max_no_improve = 300
    
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


def _multi_start_local_search(target_L, budget, seed=0):
    """Try multiple initial constructions and local search from each."""
    best_score = 0
    best_res = None
    
    limit = min(target_L + 200, 5000)
    
    for s in range(5):
        if time.time() >= DEADLINE - 5:
            break
        
        res = _initial_greedy(target_L, seed=seed + s * 1000)
        score = _verify(res, limit)
        
        if score > best_score:
            best_score = score
            best_res = list(res)
        
        if time.time() < DEADLINE - 10:
            lc_budget = min(3.0, max(0.5, (DEADLINE - time.time()) / 6))
            s2, r2 = _local_search(res, limit, budget=lc_budget, seed=seed + s * 1000 + 1)
            if s2 > best_score:
                best_score = s2
                best_res = r2
    
    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()

    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]

    # Try various target L values
    L_candidates = [200, 400, 600, 800, 1000, 1200, 1500]
    
    for L in L_candidates:
        if time.time() >= DEADLINE - 10:
            break
        
        score, res = _multi_start_local_search(L, budget=2.0, seed=2024 + L)
        if res is not None and score > best_score:
            best_score = score
            best_res = list(res)
        
        if best_score >= 1000:
            break
    
    # Final verification with extended limit
    final_score = _verify(best_res, min(best_score + 2000, MAX_L))
    if final_score > best_score:
        best_score = final_score
    
    return best_res
