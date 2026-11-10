"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using NumPy vectorized sieve
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    small_factor[0] = 0
    
    for p in small_primes:
        # For each power p^k, multiply small_factor by p at multiples of p^k
        pk = p
        while pk <= max_val:
            # Indices that are multiples of pk
            indices = np.arange(pk, max_val + 1, pk)
            small_factor[indices] *= p
            pk *= p
    
    # Convert to list for faster scalar access in the sliding window
    sf = small_factor.tolist()
    
    best_score = 25
    best_n = 1000
    
    # Sliding window with array-based frequency counting
    # Max possible value of small_factor: 2^a * 3^b * ... for numbers up to 1000024
    # 2^19 = 524288, 2^19 * 3^12 is way too big, but actual max is bounded
    # Let's use a dict for safety since values can be large
    freq = {}
    distinct = 0
    
    # Initialize window for n=1000: factors of 1001..1024
    for i in range(1, 25):
        f = sf[1000 + i]
        if f not in freq:
            freq[f] = 0
            distinct += 1
        freq[f] += 1
    
    if distinct < best_score:
        best_score = distinct
        best_n = 1000
    
    # Slide window: for n from 1001 to 1000000
    # Remove sf[n], add sf[n+24]
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove element at position n
        remove_f = sf[n]
        freq[remove_f] -= 1
        if freq[remove_f] == 0:
            del freq[remove_f]
            distinct -= 1
        
        # Add element at position n+24
        add_f = sf[n + 24]
        if add_f not in freq:
            freq[add_f] = 0
            distinct += 1
        freq[add_f] += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 1:
                break
    
    return best_n
# EVOLVE-BLOCK-END
