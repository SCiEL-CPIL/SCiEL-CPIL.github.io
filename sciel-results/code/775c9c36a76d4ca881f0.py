"""Rewritten program: local search minimizing max |sum| over all APs in [1,256]."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _max_abs_ap_sum_numpy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in indices 0..n-1 using NumPy for speed.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _local_search_fast(signs, max_time, rng):
    """
    Fast local search using the numpy evaluator.
    """
    n = len(signs)
    signs = signs.copy()
    current_max = _max_abs_ap_sum_numpy(signs)
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() < deadline:
        improved = False
        positions = list(range(n))
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline:
                break
            
            signs[pos] = -signs[pos]
            new_max = _max_abs_ap_sum_numpy(signs)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                break
            else:
                signs[pos] = -signs[pos]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                signs = best_signs.copy()
                num_flips = rng.randint(1, 4)
                for _ in range(num_flips):
                    p = rng.randint(0, n - 1)
                    signs[p] = -signs[p]
                current_max = _max_abs_ap_sum_numpy(signs)
                no_improve_count = 0
                
    return best_signs, best_max

def _legendre_init(p):
    """
    Generate a sequence of length n based on the Legendre symbol modulo p.
    For index i (0-based), we consider (i+1) mod p.
    Legendre symbol (a/p) is 1 if a is a quadratic residue mod p, -1 otherwise.
    We map 0 to 1 (arbitrary, but 0 is not a residue for p>2).
    """
    n = _N
    # Compute quadratic residues mod p
    residues = set()
    for i in range(p):
        residues.add((i * i) % p)
    
    signs = np.empty(n, dtype=np.int8)
    for i in range(n):
        val = (i + 1) % p
        if val in residues:
            signs[i] = 1
        else:
            signs[i] = -1
    return signs

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    start_time = time.time()
    total_budget = 58  # 60s limit, leave 2s margin for safety
    
    rng = random.Random(42)
    
    # Try multiple primes near 256 for initialization
    # Primes near 256: 251, 257, 263, 269, 271, 277, 281, 283
    primes_to_try = [251, 257, 263, 269, 271, 277, 281, 283]
    
    best_signs = None
    best_max = float('inf')
    
    # Phase 1: Evaluate Legendre initializations
    for p in primes_to_try:
        signs = _legendre_init(p)
        current_max = _max_abs_ap_sum_numpy(signs)
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > total_budget - 30:
            break
    
    # Also try a random initialization to have a baseline
    signs_rand = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
    rand_max = _max_abs_ap_sum_numpy(signs_rand)
    if rand_max < best_max:
        best_max = rand_max
        best_signs = signs_rand.copy()
    
    # Phase 2: Local search from the best initialization
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5 and best_signs is not None:
        result_signs, result_max = _local_search_fast(best_signs, remaining - 5, rng)
        if result_max < best_max:
            best_max = result_max
            best_signs = result_signs
    
    # Phase 3: If time remains, try local search from other good initializations
    remaining2 = total_budget - (time.time() - start_time)
    if remaining2 > 10:
        # Pick the second best initialization
        candidates = []
        for p in primes_to_try:
            signs = _legendre_init(p)
            candidates.append((signs, p))
        
        # Sort by initial discrepancy (approximate, just re-evaluate top few)
        # Actually, just pick one different from the best if possible
        # For simplicity, try one more random restart from a different prime
        if len(primes_to_try) > 1:
            # Pick a different prime
            other_prime = primes_to_try[1] if primes_to_try[0] not in [251] else primes_to_try[0]
            # Just pick the next one
            idx = 1
            signs2 = _legendre_init(primes_to_try[idx])
            result_signs2, result_max2 = _local_search_fast(signs2, remaining2 - 5, rng)
            if result_max2 < best_max:
                best_max = result_max2
                best_signs = result_signs2
    
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
