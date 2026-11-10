"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills after 90s. We aim to finish well before.
    deadline = start_time + 85
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max value we need to precompute: n+24 where n <= 1000000
    max_val = 1000024
    
    # Precompute small-prime factor for all numbers using vectorized sieve
    # sf[x] = product of prime factors p < 24 (with multiplicity)
    sf = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve for small prime factors
    for p in small_primes:
        # For each power p^k, multiply sf at indices divisible by p^k by p
        pk = p
        while pk <= max_val:
            # Vectorized multiplication
            # indices: pk, 2pk, 3pk, ...
            indices = np.arange(pk, max_val + 1, pk)
            sf[indices] *= p
            pk *= p
            
            # Check time to ensure we don't overrun during precomputation
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    # Identify rough numbers: sf[x] == 1
    # A number is rough if it has no prime factors < 24
    rough = (sf == 1).astype(np.int32)
    
    # Precompute cumulative sum of rough for O(1) window queries
    cum_rough = np.zeros(max_val + 2, dtype=np.int32)
    np.cumsum(rough, out=cum_rough[1:])
    
    window_size = 24
    min_n = 1000
    max_n = 1000000
    
    # Generate all valid n
    n_arr = np.arange(min_n, max_n + 1, dtype=np.int64)
    
    # Compute rough count for each window [n+1, n+24]
    # Window for n contains indices n+1 to n+24 in the sf array
    # cum_rough[n+24] - cum_rough[n] gives the count of 1s in that range
    rough_counts = cum_rough[n_arr + window_size] - cum_rough[n_arr]
    
    # Sort indices by rough count in descending order
    # This prioritizes windows with the most '1's, which are most likely to have low distinct counts
    sorted_indices = np.argsort(-rough_counts)
    
    best_score = window_size + 1
    best_n = min_n
    
    # Iterate through candidates
    # We can prune if the theoretical minimum distinct count for a window
    # exceeds our current best_score.
    # For a window with r roughs, the r numbers map to 1.
    # The remaining (24-r) numbers map to values != 1.
    # The minimum possible distinct values is 1 (for the 1s) + (24-r) (if all others are distinct and != 1).
    # Actually, it's possible for non-roughs to map to 1? No, by definition rough is sf==1.
    # So non-roughs map to values > 1.
    # Can non-roughs map to the same value? Yes.
    # So the absolute lower bound is 1 + (number of distinct values among non-roughs).
    # The worst case for the lower bound (optimistic for pruning) is that all non-roughs are distinct from each other and from 1.
    # So LB = 1 + (24 - r).
    # If LB >= best_score, we cannot improve, so we stop.
    
    for idx in sorted_indices:
        if time.time() > deadline - 2:
            break
            
        r = int(rough_counts[idx])
        
        # Theoretical lower bound on distinct count
        lb = 1 + (window_size - r)
        
        if lb >= best_score:
            # Since we are processing in order of decreasing r, 
            # all subsequent windows have r' <= r, so lb' >= lb >= best_score.
            # We can safely terminate.
            break
        
        n = int(n_arr[idx])
        
        # Extract the window values
        # n+1 to n+24
        window_vals = sf[n+1 : n+25]
        
        # Count distinct values
        # Using a set on the numpy array converted to list is fast for small N (24)
        distinct_count = len(set(window_vals.tolist()))
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            
            # If we found 1, it's the absolute minimum (only possible if all 24 are rough, 
            # which is rare/impossible for 24 consecutive ints, but theoretically 1 is min)
            # Actually, if best_score is 2, and lb for next is 2, we continue to see if we can get 1?
            # No, lb is the lower bound. If lb is 2, we can't get 1.
            # If we find best_score = 2, and next window has r=22, lb = 1 + 2 = 3 > 2, so we break.
            # If next window has r=23, lb = 1 + 1 = 2 == 2, we check it.
            
            if best_score == 1:
                break

    return best_n
# EVOLVE-BLOCK-END
