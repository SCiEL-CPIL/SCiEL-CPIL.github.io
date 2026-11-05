"""Rewritten program for maximizing tau_perp(n)."""
import math
import time
import numpy as np
from functools import lru_cache
from itertools import product

# EVOLVE-BLOCK-START

LIMIT = 10**18
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

def _score_from_exps(primes, exps):
    """
    Fast scoring: generate all divisors as integers, sort, count coprime adjacent pairs.
    Uses numpy for vectorized gcd where possible, but falls back to Python math.gcd.
    """
    k = len(primes)
    if k == 0:
        return 0
    total_divs = 1
    for e in exps:
        total_divs *= (e + 1)
    if total_divs < 2:
        return 0
    
    # Generate all divisors
    divisors = [1]
    for i in range(k):
        p = primes[i]
        e = exps[i]
        new_divs = []
        pw = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * pw)
            pw *= p
        divisors = new_divs
    
    divisors.sort()
    
    # Count coprime adjacent pairs
    score = 0
    n_divs = len(divisors)
    for i in range(n_divs - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            score += 1
    return score


def _score_from_exps_fast(primes, exps):
    """
    Even faster: use the fact that divisors can be generated in sorted order
    for small numbers of primes. For larger, use a heap-based approach or
    just generate and sort (which is fast enough for our sizes).
    """
    return _score_from_exps(primes, exps)


def _factorize(n):
    """Factorize n into {prime: exponent}."""
    if n < 2:
        return {}
    factors = {}
    x = n
    for p in PRIMES:
        if p * p > x:
            break
        while x % p == 0:
            factors[p] = factors.get(p, 0) + 1
            x //= p
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors


def _exps_to_n(primes, exps):
    """Convert primes and exponents to n."""
    n = 1
    for i, p in enumerate(primes):
        if exps[i] > 0:
            n *= p ** exps[i]
    return n


def _beam_search(num_primes, beam_width=50, max_iters=2000, deadline=None):
    """
    Beam search over exponent assignments for the first num_primes primes.
    State: tuple of exponents.
    We start with all 1s and try to increase exponents greedily.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    primes = PRIMES[:num_primes]
    
    # Initial state: all exponents = 1
    init_exps = [1] * num_primes
    init_n = 1
    for p in primes:
        init_n *= p
    if init_n > LIMIT:
        return None, None
    
    # Beam: list of (score, exps_tuple)
    init_score = _score_from_exps(primes, init_exps)
    beam = [(init_score, tuple(init_exps))]
    best_score = init_score
    best_exps = init_exps[:]
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        if not beam:
            break
        
        # Expand each state in the beam
        candidates = []
        for score, exps_t in beam:
            exps_list = list(exps_t)
            n = _exps_to_n(primes, exps_list)
            
            # Try increasing each exponent
            for i in range(num_primes):
                new_exp = exps_list[i] + 1
                new_n = n * primes[i]
                if new_n > LIMIT:
                    continue
                exps_list[i] = new_exp
                s = _score_from_exps(primes, exps_list)
                candidates.append((s, tuple(exps_list)))
                exps_list[i] = new_exp - 1  # restore
            
            # Try decreasing each exponent (to explore different regions)
            for i in range(num_primes):
                if exps_list[i] > 0:
                    new_exp = exps_list[i] - 1
                    exps_list[i] = new_exp
                    n_new = _exps_to_n(primes, exps_list)
                    s = _score_from_exps(primes, exps_list)
                    candidates.append((s, tuple(exps_list)))
                    exps_list[i] = new_exp + 1  # restore
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: -x[0])
        seen = set()
        new_beam = []
        for s, e in candidates:
            if e not in seen:
                seen.add(e)
                new_beam.append((s, e))
                if len(new_beam) >= beam_width:
                    break
        
        beam = new_beam
        
        # Update best
        if beam and beam[0][0] > best_score:
            best_score = beam[0][0]
            best_exps = list(beam[0][1])
    
    if best_exps is None:
        return None, None
    
    best_n = _exps_to_n(primes, best_exps)
    if best_n > LIMIT or best_n < 2:
        return None, None
    
    return best_n, best_score


def _random_restart_search(num_primes, n_restarts=100, deadline=None):
    """
    Random restart local search: randomly initialize exponents, then do hill climbing.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    primes = PRIMES[:num_primes]
    best_n = None
    best_score = 0
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization: exponents between 1 and some max
        # Bias toward smaller exponents for larger primes
        exps = []
        n = 1
        for i, p in enumerate(primes):
            # Max exponent such that p^e * n <= LIMIT
            max_e = 1
            while n * p ** (max_e + 1) <= LIMIT:
                max_e += 1
            # Random choice, biased toward 1-2
            import random
            if max_e <= 2:
                e = random.randint(1, max_e)
            else:
                # Geometric-like distribution
                if random.random() < 0.5:
                    e = 1
                elif random.random() < 0.7:
                    e = 2
                else:
                    e = random.randint(1, min(max_e, 5))
            exps.append(e)
            n *= p ** e
        
        if n > LIMIT:
            continue
        
        # Hill climbing
        current_exps = exps[:]
        current_n = n
        current_score = _score_from_exps(primes, current_exps)
        
        improved = True
        max_steps = 50
        steps = 0
        while improved and steps < max_steps:
            improved = False
            steps += 1
            for i in range(num_primes):
                # Try increasing
                old_e = current_exps[i]
                new_e = old_e + 1
                new_n = current_n * primes[i]
                if new_n <= LIMIT:
                    current_exps[i] = new_e
                    s = _score_from_exps(primes, current_exps)
                    if s > current_score:
                        current_score = s
                        current_n = new_n
                        improved = True
                    else:
                        current_exps[i] = old_e
                # Try decreasing
                if old_e > 1:
                    new_e = old_e - 1
                    new_n = current_n // primes[i]
                    current_exps[i] = new_e
                    s = _score_from_exps(primes, current_exps)
                    if s > current_score:
                        current_score = s
                        current_n = new_n
                        improved = True
                    else:
                        current_exps[i] = old_e
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
    
    return best_n, best_score


def _try_structured_candidates(deadline=None):
    """
    Try a set of structured candidates: primorials with varied exponents.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    best_n = None
    best_score = 0
    
    # Primorial of first k primes
    for k in range(2, 16):
        if time.time() > deadline:
            break
        primes_k = PRIMES[:k]
        n = 1
        for p in primes_k:
            n *= p
        if n > LIMIT:
            break
        s = _score_from_exps(primes_k, [1] * k)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Try various exponent patterns
    patterns = [
        # (num_primes, list of exponents)
        (15, [1]*15),
        (14, [2]+[1]*13),
        (14, [1,2]+[1]*12),
        (13, [3]+[1]*12),
        (13, [2,2]+[1]*11),
        (12, [4]+[1]*11),
        (12, [3,2]+[1]*10),
        (12, [2,2,2]+[1]*9),
        (11, [5]+[1]*10),
        (11, [4,2]+[1]*9),
        (11, [3,3]+[1]*9),
        (11, [3,2,2]+[1]*8),
        (10, [6]+[1]*9),
        (10, [5,2]+[1]*8),
        (10, [4,3]+[1]*8),
        (10, [4,2,2]+[1]*7),
        (10, [3,3,2]+[1]*7),
        (9, [7]+[1]*8),
        (9, [6,2]+[1]*7),
        (9, [5,3]+[1]*7),
        (9, [5,2,2]+[1]*6),
        (9, [4,4]+[1]*6),
        (9, [4,3,2]+[1]*5),
        (8, [8]+[1]*7),
        (8, [7,2]+[1]*6),
        (8, [6,3]+[1]*6),
        (8, [6,2,2]+[1]*5),
        (8, [5,4]+[1]*5),
        (8, [5,3,2]+[1]*4),
        (8, [4,4,2]+[1]*4),
        (7, [9]+[1]*6),
        (7, [8,2]+[1]*5),
        (7, [7,3]+[1]*5),
        (7, [7,2,2]+[1]*4),
        (7, [6,4]+[1]*4),
        (7, [6,3,2]+[1]*3),
        (6, [10]+[1]*5),
        (6, [9,2]+[1]*4),
        (6, [8,3]+[1]*4),
        (6, [8,2,2]+[1]*3),
        (6, [7,4]+[1]*3),
        (6, [7,3,2]+[1]*2),
        (5, [11]+[1]*4),
        (5, [10,2]+[1]*3),
        (5, [9,3]+[1]*3),
        (5, [9,2,2]+[1]*2),
        (5, [8,4]+[1]*2),
        (4, [12]+[1]*3),
        (4, [11,2]+[1]*2),
        (4, [10,3]+[1]*2),
        (4, [10,2,2]),
        (4, [9,4]+[1]),
        (3, [13]+[1]*2),
        (3, [12,2]+[1]),
        (3, [11,3]),
        (3, [11,2,2]),
        (2, [14]+[1]),
        (2, [13,2]),
        (2, [12,3]),
        (2, [12,2,2]),
    ]
    
    for num_p, exps in patterns:
        if time.time() > deadline:
            break
        if num_p > len(PRIMES):
            continue
        primes_k = PRIMES[:num_p]
        if len(exps) != num_p:
            continue
        n = _exps_to_n(primes_k, exps)
        if n > LIMIT or n < 2:
            continue
        s = _score_from_exps(primes_k, exps)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 2
    best_score = 0
    
    # Phase 1: Structured candidates (fast)
    n1, s1 = _try_structured_candidates(deadline=deadline)
    if n1 is not None and s1 > best_score:
        best_n = n1
        best_score = s1
    
    if time.time() > deadline - 5:
        return best_n
    
    # Phase 2: Beam search for various numbers of primes
    for num_p in [15, 14, 13, 12, 11, 10, 9, 8]:
        if time.time() > deadline - 10:
            break
        n2, s2 = _beam_search(num_p, beam_width=30, max_iters=500, deadline=deadline)
        if n2 is not None and s2 > best_score:
            best_n = n2
            best_score = s2
    
    if time.time() > deadline - 5:
        return best_n
    
    # Phase 3: Random restart local search
    for num_p in [15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
        n3, s3 = _random_restart_search(num_p, n_restarts=50, deadline=deadline)
        if n3 is not None and s3 > best_score:
            best_n = n3
            best_score = s3
    
    return best_n

# EVOLVE-BLOCK-END
