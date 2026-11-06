import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    def greedy_randomized():
        S = []
        S_bits = 0
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        C2_bits = 0
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check validity
            # 1. Check sums x+y for y in S
            # new_sums_bits represents the set {x+y | y in S}
            new_sums_bits = S_bits << x
            if new_sums_bits & C2_bits:
                continue
            
            # 2. Check sum 2x
            if counts[2 * x] >= 2:
                continue
            
            # Valid, add x
            S.append(x)
            S_bits |= (1 << x)
            
            # Update counts
            # We need to iterate through S to update counts for x+y
            # Note: S does not contain x yet in the loop below if we iterate over the list?
            # We just appended x. So we should iterate over S[:-1] for y, and handle 2x separately.
            # Or iterate over all y in S including x? 
            # If we iterate over all y in S (including x), we get x+x=2x and x+y for y<x.
            # This covers all new pairs (x, y) with y in S_new.
            # Since we are adding x, the new pairs are (x, y) for all y in S_new.
            # This is correct.
            
            for y in S:
                s = x + y
                counts[s] += 1
                if counts[s] == 2:
                    C2_bits |= (1 << s)
                    
        return S, S_bits, counts, C2_bits

    best_S, best_bits, best_counts, best_C2 = greedy_randomized()
    best_size = len(best_S)
    
    # Local Search: Try to improve
    # We can try to add elements that were not added.
    # Or try to swap.
    # Given the time, multiple randomized greedy runs might be better than complex local search.
    
    while time.time() - start_time < time_limit:
        S, bits, counts, C2 = greedy_randomized()
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            best_bits = bits
            best_counts = counts
            best_C2 = C2
            
    return best_S

# EVOLVE-BLOCK-END
