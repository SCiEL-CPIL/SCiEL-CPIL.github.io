import time
import math
import random
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate powerful numbers up to L
    L = 10**13
    # Primes for square-free check up to sqrt(max_b)
    # max_b is approx L**(1/3) = 10**(13/3) ~ 21544
    # sqrt(21544) ~ 147
    # Let's just check divisibility by squares of primes up to 200
    
    def is_square_free(n):
        if n == 1:
            return True
        # Check primes up to sqrt(n)
        # Simple check: for k in 2..sqrt(n), if k*k divides n, not square free
        # Optimize: only check primes
        # But n is small, simple loop is fine
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    powerful_set = set()
    max_b = int(L ** (1/3)) + 2
    
    # Generate
    for b in range(1, max_b + 1):
        if time.time() > deadline - 60: # Reserve 60s for search
            break
        if not is_square_free(b):
            continue
        b3 = b ** 3
        if b3 > L:
            break
        max_a = int((L // b3) ** 0.5)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerful_set.add(val)
            else:
                break

    powerful_list = sorted(powerful_set)
    n_powerful = len(powerful_list)
    
    if n_powerful == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    # 2. Generate Candidates
    candidates = set()
    
    # Heuristic: Sums of small powerful numbers
    # Use NumPy for speed
    M = min(5000, n_powerful)
    P_sub = np.array(powerful_list[:M], dtype=np.int64)
    
    # Compute pairwise sums
    # To save memory, we can do it in chunks or just accept 5000x5000
    # 5000*5000*8 bytes = 200MB. Acceptable.
    sums = np.add.outer(P_sub, P_sub)
    
    # We only care about sums <= 2*L
    # All sums are <= 2 * max(P_sub) which is <= 2 * powerful_list[M-1]
    # This is likely much smaller than 2*L if M is small, but let's filter anyway
    # Actually, we want to find N that is a sum of TWO powerful numbers from the FULL set.
    # The heuristic is that frequent sums in the small subset are likely frequent in the full set.
    
    # Find most frequent sums
    # Flatten and count
    flat_sums = sums.flatten()
    # Filter sums that are valid (<= 2*L)
    valid_sums = flat_sums[flat_sums <= 2 * L]
    
    # Use np.unique for counting
    unique_sums, counts = np.unique(valid_sums, return_counts=True)
    
    # Get top 20 candidates
    if len(counts) > 0:
        top_indices = np.argsort(counts)[-20:]
        for idx in top_indices:
            candidates.add(int(unique_sums[idx]))
    
    # Add random candidates
    # Range: [10**11, 2*L]
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        n = random.randint(10**11, 2 * L)
        candidates.add(n)
        
    # 3. Evaluate Candidates
    best_n = -1
    best_count = -1
    best_pairs = []
    
    # Sort candidates to process
    for n in candidates:
        if time.time() > deadline - 10:
            break
            
        # Count pairs (a, b) with a <= b, a+b=n, a,b in powerful_set
        # Iterate a in powerful_list where a <= n/2
        # Use bisect to find range of a
        # a <= n/2
        # b = n - a >= a
        
        count = 0
        pairs = []
        
        # Efficient iteration
        # We can just iterate through powerful_list until a > n/2
        # powerful_list is sorted
        half_n = n // 2
        
        # Find index where powerful_list > half_n
        # Use bisect
        import bisect
        right_idx = bisect.bisect_right(powerful_list, half_n)
        
        for i in range(right_idx):
            a = powerful_list[i]
            b = n - a
            if b < a:
                break
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
                
        if count > best_count:
            best_count = count
            best_n = n
            best_pairs = pairs

    if best_n == -1:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_n, 'pairs': best_pairs}
