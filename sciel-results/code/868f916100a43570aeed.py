import time
import random
from itertools import combinations

def _prime_masks():
    MAX = 256
    is_prime = [True] * (MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX + 1, i):
                is_prime[j] = False
    
    odd_primes = [p for p in range(3, MAX + 1) if is_prime[p]]
    prime_to_bit = {p: i for i, p in enumerate(odd_primes)}
    
    masks = [0] * (MAX + 1)
    for n in range(1, MAX + 1):
        if n % 2 == 0:
            continue
        mask = 0
        for p in odd_primes:
            if p > n:
                break
            if n % p == 0:
                mask |= (1 << prime_to_bit[p])
        masks[n] = mask
    return masks

def _pairwise_coprime_masks(masks_list):
    n = len(masks_list)
    for i in range(n):
        for j in range(i+1, n):
            if masks_list[i] & masks_list[j]:
                return False
    return True

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    masks = _prime_masks()
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Phase 1: Random Search (fast, might find 9 quickly)
    # We prioritize finding 9 distinct sums.
    start_time = time.time()
    while time.time() - start_time < 5:
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        count = len(distinct_sums)
        
        if count <= best_count:
            continue
            
        sum_masks = [masks[s] for s in distinct_sums]
        if _pairwise_coprime_masks(sum_masks):
            best_count = count
            best_A = A
            best_B = B
            
            if best_count == 9:
                return [best_A, best_B]

    # Phase 2: Exhaustive Search (if 9 not found, maximize count)
    # To speed this up, we only iterate if we haven't found 9.
    # We iterate over A combinations and B combinations.
    # Given the space is large, we sample A combinations and do full B search for each?
    # Or just continue random search?
    
    # If 9 is achievable, random search is likely to find it within seconds if the solution space isn't tiny.
    # If 9 is not achievable, we want the max (likely 8 or 7).
    
    # Let's do a limited exhaustive search.
    # We'll iterate over a subset of A combinations.
    # Number of A combos: 41664.
    # We can check all A combos, and for each, sample B combos?
    
    # Strategy: Iterate all A. For each A, iterate all B? Too slow.
    # Strategy: Iterate all A. For each A, do a local search/random search for B?
    
    # Let's just extend the random search with more iterations if time permits,
    # but also try a structured scan on a subset of "good" A's.
    
    # "Good" A's: those with large gaps, to encourage distinct sums.
    # Sort odds by value?
    
    # Simpler: Just run random search for the remaining time, keeping the best.
    # But we might miss a non-random 9-solution.
    
    # Let's try to iterate A, and for each A, try a few B's.
    # Or iterate B, and for each B, try a few A's.
    
    # Given the previous program found 1.8 (likely 9), and it did a full loop (broken by time),
    # it implies 9 might be found relatively early or by luck.
    
    # We will continue random search for the rest of the time.
    while time.time() < deadline - 5:
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        count = len(distinct_sums)
        
        if count <= best_count:
            continue
            
        sum_masks = [masks[s] for s in distinct_sums]
        if _pairwise_coprime_masks(sum_masks):
            best_count = count
            best_A = A
            best_B = B
            if best_count == 9:
                return [best_A, best_B]

    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
