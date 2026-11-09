"""Optimize squarefree n maximizing coprime adjacent divisor pairs."""
# EVOLVE-BLOCK-START
import time
import math
import random

DEADLINE = 145.0
MAX_N = 10**18

def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

# Generate a large pool of primes
PRIMES = _sieve(20000)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

def _product(primes):
    r = 1
    for p in primes:
        r *= p
        if r > MAX_N:
            return r
    return r

def _score(primes):
    """Calculate tau_perp score for a set of primes."""
    divs = [1]
    for p in primes:
        # Append p * d for all existing d
        divs += [d * p for d in divs]
    divs.sort()
    c = 0
    # Optimization: local variable lookup
    gcd = math.gcd
    prev = divs[0]
    for x in divs[1:]:
        if gcd(prev, x) == 1:
            c += 1
        prev = x
    return c

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_score = -1
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    eval_cache = {}

    def get_score(primes_tuple):
        if primes_tuple in eval_cache:
            return eval_cache[primes_tuple]
        n = 1
        for p in primes_tuple:
            n *= p
            if n > MAX_N:
                eval_cache[primes_tuple] = -1
                return -1
        s = _score(list(primes_tuple))
        eval_cache[primes_tuple] = s
        return s

    def update_best(primes_list):
        nonlocal best_score, best_n, best_primes
        n = 1
        for p in primes_list:
            n *= p
        if n <= MAX_N:
            s = get_score(tuple(sorted(primes_list)))
            if s > best_score:
                best_score = s
                best_n = n
                best_primes = sorted(primes_list)

    # Phase 1: Determine best k
    # Try k from 2 to 16
    best_k = 8
    for k in range(2, 17):
        if time.time() - start > DEADLINE - 120:
            break
        cand = tuple(PRIMES[:k])
        n = 1
        for p in cand:
            n *= p
        if n > MAX_N:
            break
        s = get_score(cand)
        if s > best_score:
            best_score = s
            best_k = k
            best_n = n
            best_primes = list(cand)

    # Phase 2: Local Search for best_k and neighbors
    # We will run local search for k, k-1, k+1
    for k in [max(2, best_k - 1), best_k, min(16, best_k + 1)]:
        if time.time() - start > DEADLINE - 30:
            break
        
        # Initialize with smallest k primes
        cur = list(PRIMES[:k])
        cur_set = set(cur)
        cur_score = get_score(tuple(cur))
        
        # If we already have a better score from Phase 1 for this k, use that?
        # Phase 1 only checked smallest k. So we start there.
        
        improved = True
        while improved and (time.time() - start < DEADLINE - 20):
            improved = False
            # Try to swap each prime with a better one
            for i in range(k):
                if time.time() - start > DEADLINE - 20:
                    break
                p_old = cur[i]
                idx_old = PRIME_TO_IDX[p_old]
                
                # Candidate primes: 
                # 1. Primes slightly larger (to increase n but maybe improve structure)
                # 2. Primes slightly smaller (to decrease n)
                # We search a window around the current index.
                # The window should be wide enough to escape local minima.
                lo = max(0, idx_old - 50)
                hi = min(len(PRIMES), idx_old + 500)
                
                best_swap_p = None
                best_swap_score = cur_score
                
                for j in range(lo, hi):
                    p_new = PRIMES[j]
                    if p_new in cur_set:
                        continue
                    
                    new_n = _product(cur[:i] + [p_new] + cur[i+1:])
                    if new_n > MAX_N:
                        continue
                    
                    new_tuple = tuple(sorted(cur[:i] + [p_new] + cur[i+1:]))
                    s = get_score(new_tuple)
                    
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap_p = p_new
                
                if best_swap_p is not None:
                    cur[i] = best_swap_p
                    cur_set.remove(p_old)
                    cur_set.add(best_swap_p)
                    cur_score = best_swap_score
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_n = _product(cur)
                        best_primes = sorted(cur)
            
        # Phase 3: Random Restarts for the current k
        if time.time() - start < DEADLINE - 10:
            for restart in range(50):
                if time.time() - start > DEADLINE - 5:
                    break
                
                # Create a random initial set
                # Mix of small primes and some random ones
                if rng.random() < 0.5:
                    # Start from current best for this k
                    cur = list(best_primes) if len(best_primes) == k else list(PRIMES[:k])
                else:
                    # Random start
                    # Pick k primes, biased towards smaller ones
                    indices = []
                    # Ensure we don't pick duplicates
                    # We can just pick random indices and resolve collisions
                    candidates = list(range(0, 100)) # First 100 primes
                    rng.shuffle(candidates)
                    cur = [PRIMES[i] for i in candidates[:k]]
                    if _product(cur) > MAX_N:
                        cur = list(PRIMES[:k])
                        if _product(cur) > MAX_N:
                            continue
                
                cur_set = set(cur)
                cur_score = get_score(tuple(cur))
                
                # Quick local search
                for it in range(20):
                    if time.time() - start > DEADLINE - 5:
                        break
                    i = rng.randrange(k)
                    p_old = cur[i]
                    idx_old = PRIME_TO_IDX[p_old]
                    lo = max(0, idx_old - 20)
                    hi = min(len(PRIMES), idx_old + 200)
                    
                    best_swap_p = None
                    best_swap_score = cur_score
                    
                    # Sample a few candidates to speed up
                    sample_size = min(50, hi - lo)
                    sample_indices = rng.sample(range(lo, hi), sample_size)
                    
                    for j in sample_indices:
                        p_new = PRIMES[j]
                        if p_new in cur_set:
                            continue
                        new_n = _product(cur[:i] + [p_new] + cur[i+1:])
                        if new_n > MAX_N:
                            continue
                        new_tuple = tuple(sorted(cur[:i] + [p_new] + cur[i+1:]))
                        s = get_score(new_tuple)
                        if s > best_swap_score:
                            best_swap_score = s
                            best_swap_p = p_new
                    
                    if best_swap_p is not None:
                        cur[i] = best_swap_p
                        cur_set.remove(p_old)
                        cur_set.add(best_swap_p)
                        cur_score = best_swap_score
                        if cur_score > best_score:
                            best_score = cur_score
                            best_n = _product(cur)
                            best_primes = sorted(cur)

    return best_n
# EVOLVE-BLOCK-END
