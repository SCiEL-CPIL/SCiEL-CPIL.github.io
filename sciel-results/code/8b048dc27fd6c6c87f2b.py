"""Hybrid Greedy-then-Anneal solver for Erdos #689 (n=1000) doubly-covered run.

Structural approach:
1. Greedy construction: assign residues to small primes to doubly cover 1..L.
   Process x in order, assign the smallest available primes that can cover x.
2. Gap repair: identify the first gap and try to fix it by reassigning residues.
3. Simulated annealing: cost-based moves that target under-covered integers.
   The cost function is the number of integers in 1..L with coverage < 2.
"""
import time
import sys
import random
import numpy as np

T_START = time.time()
DEADLINE = 118.0
MAX_L = 100000
sys.setrecursionlimit(100000)


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)
_BY_P = {p: i for i, p in enumerate(_PRIMES)}
_N_PRIMES = len(_PRIMES)
_SMALL_IDX = [i for i, p in enumerate(_PRIMES) if p <= 10000]
_SMALL_PRIMES = [_PRIMES[i] for i in _SMALL_IDX]


def _verify(residues, limit=MAX_L):
    """Find largest L such that every x in 1..L has >= 2 covering primes."""
    p = np.asarray(_PRIMES, dtype=np.int64)
    r = np.asarray(residues, dtype=np.int64)
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(p, r):
        rr = int(rr)
        start = rr
        if start < 1:
            start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    ok = mark[1:] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _compute_coverage(residues, limit):
    """Compute coverage count for each x in 1..limit."""
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, residues):
        rr = int(rr)
        start = rr
        if start < 1:
            start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    return mark


def _greedy_construct(L, seed=0):
    """Greedy construction: assign residues to doubly cover 1..L."""
    rng = random.Random(seed)
    residues = [-1] * _N_PRIMES
    small_primes = [p for p in _PRIMES if p <= L]
    
    # Process x from 1 to L
    for x in range(1, L + 1):
        # Count how many assigned primes cover x
        count = 0
        for p in small_primes:
            pi = _BY_P[p]
            r = residues[pi]
            if r >= 0 and x % p == r:
                count += 1
        
        if count >= 2:
            continue
        
        need = 2 - count
        
        # Find unassigned small primes
        candidates = [p for p in small_primes if residues[_BY_P[p]] < 0]
        
        if len(candidates) < need:
            return None
        
        # Prefer smaller primes (they cover more frequently)
        candidates.sort()
        
        # Assign the smallest 'need' primes
        for p in candidates[:need]:
            residues[_BY_P[p]] = x % p
    
    # Fill remaining residues with 0
    full_res = [0] * _N_PRIMES
    for i in range(_N_PRIMES):
        if residues[i] >= 0:
            full_res[i] = residues[i]
    return full_res


def _gap_repair(residues, L, budget=2.0):
    """Try to fix the first gap by reassigning residues."""
    start_time = time.time()
    rng = random.Random(int(time.time() * 1000))
    
    for _ in range(100):
        if time.time() - start_time > budget:
            break
        
        # Find the first gap
        mark = _compute_coverage(residues, L + 100)
        ok = mark[1:] >= 2
        if ok.all():
            return L + 100, residues
        
        first_gap = int(np.argmax(~ok))
        
        # Try to fix by changing a random small prime's residue
        small_primes = [p for p in _PRIMES if p <= L]
        p = rng.choice(small_primes)
        pi = _BY_P[p]
        
        # Try a few random residues
        for _ in range(10):
            new_r = rng.randint(0, p - 1)
            trial = list(residues)
            trial[pi] = new_r
            
            # Check if this fixes the gap
            mark2 = _compute_coverage(trial, L + 100)
            ok2 = mark2[1:] >= 2
            if ok2.all():
                return L + 100, trial
            
            # Check if this improves the score
            if ok2[:first_gap].all() and not ok[:first_gap].all():
                residues = trial
                break
    
    # Return the best we found
    mark = _compute_coverage(residues, L + 100)
    ok = mark[1:] >= 2
    if ok.all():
        return L + 100, residues
    first = int(np.argmax(~ok))
    return first, residues


def _simulated_anneal(residues, base_score, budget=5.0):
    """Simulated annealing with cost-based moves targeting under-covered integers."""
    start_time = time.time()
    rng = random.Random(int(time.time() * 1000))
    
    best_score = base_score
    best_res = list(residues)
    current_res = list(residues)
    current_score = base_score
    
    # Start temperature
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < budget:
        # Pick a random small prime
        small_primes = [p for p in _PRIMES if p <= best_score + 100]
        if not small_primes:
            small_primes = _PRIMES[:10]
        
        p = rng.choice(small_primes)
        pi = _BY_P[p]
        
        # Pick a new residue
        new_r = rng.randint(0, p - 1)
        old_r = current_res[pi]
        
        if new_r == old_r:
            continue
        
        # Compute the cost change
        # Cost = number of x in 1..(best_score+200) with coverage < 2
        # We approximate by checking the affected positions
        
        trial = list(current_res)
        trial[pi] = new_r
        
        # Quick check: compute coverage for a window around the current score
        check_limit = min(best_score + 200, MAX_L)
        mark_old = _compute_coverage(current_res, check_limit)
        mark_new = _compute_coverage(trial, check_limit)
        
        cost_old = np.sum(mark_old[1:] < 2)
        cost_new = np.sum(mark_new[1:] < 2)
        
        delta_cost = cost_new - cost_old
        
        # Accept if cost improves, or with probability exp(-delta_cost / T)
        if delta_cost <= 0:
            current_res = trial
            # Verify the actual score
            score = _verify(current_res, check_limit)
            if score > best_score:
                best_score = score
                best_res = list(current_res)
        else:
            prob = np.exp(-delta_cost / T)
            if rng.random() < prob:
                current_res = trial
                score = _verify(current_res, check_limit)
                if score > best_score:
                    best_score = score
                    best_res = list(current_res)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_score, best_res


def _hill_climb(residues, base_score, budget=2.0):
    """Try to improve by adjusting some residues."""
    best_score = base_score
    best_res = list(residues)
    end_time = time.time() + budget
    rng = random.Random(int(time.time() * 1000))
    
    while time.time() < end_time:
        idx = rng.randint(0, _N_PRIMES - 1)
        p = _PRIMES[idx]
        old_r = best_res[idx]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        trial = list(best_res)
        trial[idx] = new_r
        s = _verify(trial, base_score + 2000)
        if s > best_score:
            best_score = s
            best_res = trial
    
    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()
    rng = random.Random(20240607)
    
    # Start with the seed solution
    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]
    
    # Try greedy construction for increasing L
    for target_L in [100, 200, 500, 1000, 2000, 5000]:
        if time.time() > DEADLINE - 10:
            break
        
        # Try multiple seeds for the greedy construction
        for seed in range(3):
            if time.time() > DEADLINE - 10:
                break
            
            res = _greedy_construct(target_L, seed=seed)
            if res:
                score = _verify(res, target_L + 5000)
                if score > best_score:
                    best_score = score
                    best_res = res
                    
                    # Gap repair
                    if time.time() < DEADLINE - 15:
                        s_rep, r_rep = _gap_repair(best_res, best_score, budget=min(2.0, DEADLINE - time.time() - 15))
                        if s_rep > best_score:
                            best_score = s_rep
                            best_res = r_rep
                    
                    # Simulated annealing
                    if time.time() < DEADLINE - 10:
                        s_sa, r_sa = _simulated_anneal(best_res, best_score, budget=min(3.0, DEADLINE - time.time() - 10))
                        if s_sa > best_score:
                            best_score = s_sa
                            best_res = r_sa
                    
                    # Hill climb
                    if time.time() < DEADLINE - 5:
                        s_hc, r_hc = _hill_climb(best_res, best_score, budget=min(2.0, DEADLINE - time.time() - 5))
                        if s_hc > best_score:
                            best_score = s_hc
                            best_res = r_hc
                    
                    break  # Found a good solution, try next L
    
    # Also try the simple heuristics as fallback
    if time.time() < DEADLINE - 5:
        res1 = [0] * _N_PRIMES
        s1 = _verify(res1, 5000)
        if s1 > best_score:
            best_score = s1
            best_res = res1
        
        res2 = [1 % p for p in _PRIMES]
        s2 = _verify(res2, 5000)
        if s2 > best_score:
            best_score = s2
            best_res = res2
    
    # Final simulated annealing + hill climb
    if time.time() < DEADLINE - 8:
        s_sa, r_sa = _simulated_anneal(best_res, best_score, budget=min(4.0, DEADLINE - time.time() - 8))
        if s_sa > best_score:
            best_score = s_sa
            best_res = r_sa
    
    if time.time() < DEADLINE - 3:
        s_hc, r_hc = _hill_climb(best_res, best_score, budget=min(3.0, DEADLINE - time.time() - 3))
        if s_hc > best_score:
            best_score = s_hc
            best_res = r_hc
    
    return best_res
