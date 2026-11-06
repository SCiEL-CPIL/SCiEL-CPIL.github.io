import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n
    # sums_mask: boolean array indicating which sums are already used
    # Size max_sum + 1 to index by sum value
    
    best_set = []
    
    # We will perform multiple randomized greedy restarts.
    # Each restart tries to build a set by picking candidates in a random order.
    
    # To improve diversity, we can also vary the range or bias,
    # but simple shuffling is a strong baseline.
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Create a list of candidates and shuffle
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        sums_mask = np.zeros(max_sum + 1, dtype=bool)
        
        for x in candidates:
            # Check time inside the loop as well, as a single restart might be long
            if time.time() > deadline:
                break
            
            # Generate all new sums created by adding x to S
            new_sums = []
            
            # Triple (x, x, x)
            new_sums.append(3 * x)
            
            if S:
                S_arr = np.array(S)
                
                # Triples (x, x, a) where a in S
                # Sums: 2x + a
                new_sums.extend((2 * x + S_arr).tolist())
                
                # Triples (x, a, b) where a, b in S, a <= b
                # We need to generate all unique pairs (a, b) from S
                if len(S) > 1:
                    # Use triu_indices for upper triangle (including diagonal)
                    i, j = np.triu_indices(len(S))
                    pair_sums = S_arr[i] + S_arr[j]
                    new_sums.extend((x + pair_sums).tolist())
                elif len(S) == 1:
                    # Only pair is (S[0], S[0])
                    new_sums.append(x + 2 * S[0])
            
            # Convert to numpy array for fast checking
            new_sums_arr = np.array(new_sums)
            
            # 1. Check if any new sum collides with existing sums
            # sums_mask[new_sums_arr] will be True if any of these sums are already used
            if np.any(sums_mask[new_sums_arr]):
                continue
            
            # 2. Check if any new sums collide with each other
            # If there are duplicates in new_sums, the set is invalid
            # Using set() is efficient for small lists (size ~ |S|^2/2)
            if len(new_sums) != len(set(new_sums)):
                continue
            
            # If we reach here, x is a valid addition
            S.append(x)
            sums_mask[new_sums_arr] = True
            
            # Optional: Early exit if we reach a theoretical upper bound?
            # The bound is ~42. If we reach 42, we can stop this restart.
            # But we might find a better order in another restart?
            # No, if we found a valid set of size 42, it's likely optimal or near-optimal.
            # But let's just let it run to fill up as much as possible.
        
        # Update best set if current is larger
        if len(S) > len(best_set):
            best_set = S.copy()
            
            # If we found a very large set, we might want to spend more time
            # trying to improve it, or just continue with other restarts.
            # For now, just continue.
            
    return best_set
