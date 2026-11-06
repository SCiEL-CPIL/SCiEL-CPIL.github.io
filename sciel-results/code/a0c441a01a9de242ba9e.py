"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pows = set([1])
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # We iterate over square-free b and all a.
    max_b = int(limit ** (1/3)) + 1
    
    # Precompute square-free numbers up to max_b
    # A number n is square-free if it is not divisible by any p^2 for prime p.
    # We can just check divisibility by squares of primes <= sqrt(b).
    # For small max_b (like 10^3 or 10^4), this is fast.
    
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        
        b_cubed = b ** 3
        if b_cubed > limit:
            break
            
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            n = a * a * b_cubed
            if n <= limit:
                pows.add(n)
            else:
                break
                
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # We will generate powerful numbers up to a limit that allows O(K^2) to be fast.
    # Limit 10^7 -> K ~ 8000 -> K^2/2 ~ 3e7 ops. Fast.
    # Limit 10^8 -> K ~ 25000 -> K^2/2 ~ 3e8 ops. Slow in Python.
    # Let's try 10^7 first.
    
    limit = 10**7
    powerful_list = generate_powerful(limit)
    powerful_set = set(powerful_list)
    num_powerful = len(powerful_list)
    
    # Count representations for each N using pairs from this list
    rep_count = defaultdict(int)
    
    # Optimization: Only consider pairs (a, b) with a <= b.
    # Since the list is sorted, for each a at index i, b must be at index j >= i.
    
    for i in range(num_powerful):
        a = powerful_list[i]
        # Early break if 2*a > max_possible_N? 
        # We don't have a fixed max N, but N = a+b.
        # If we want to find the *best* N, we should consider all sums.
        
        for j in range(i, num_powerful):
            b = powerful_list[j]
            N = a + b
            rep_count[N] += 1
            
        # Time check
        if time.time() > deadline - 10:
            break
            
    if not rep_count:
        # Fallback if something went wrong or list was empty
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find the N with the maximum count
    best_N = max(rep_count.keys(), key=lambda n: rep_count[n])
    best_count = rep_count[best_N]
    
    # Collect the pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
            
    # Verify count matches (sanity check)
    # If there's a mismatch, it might be due to duplicates in powerful_list? 
    # No, powerful_list is a set converted to sorted list, so unique.
    # The count in rep_count is exact for the subset of pairs considered.
    # Since we broke the loop early if time ran out, rep_count might be incomplete.
    # But we only broke the outer loop. The inner loop finished for each i processed.
    # So rep_count is a valid count for the subset of a's processed.
    # This is a certified lower bound for that N using the processed a's.
    # However, if we didn't process all a's, we might have missed pairs for best_N.
    # To be safe and ensure 'pairs' matches the count returned (which is the score),
    # we should only return pairs we actually verified.
    # The score is the number of pairs in the returned list.
    # So we just return the pairs we found.
    
    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
