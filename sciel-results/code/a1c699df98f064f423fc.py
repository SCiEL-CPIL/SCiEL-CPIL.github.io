# EVOLVE-BLOCK-START
from itertools import combinations
import random
import time

def construct():
    # Generate all 4-subsets as integers
    # Bit i corresponds to element i+1
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # Precompute intersection matrix
    # inter[i][j] is the intersection mask of all_sets[i] and all_sets[j]
    # To save memory and time, we can compute on the fly or use a dict?
    # 2184x2184 is ~4.7M entries. 
    # A list of lists of ints is fine.
    # But initialization takes time.
    # Let's see if we can avoid full precomputation.
    # If we just use `a & b`, it's very fast in Python.
    # Let's try WITHOUT precomputation first, just using ints.
    
    best_size = 0
    best_chosen = []
    
    rng = random.Random(42)
    
    # We will run multiple trials.
    # Estimate time per trial with ints.
    # Inner loop: a & s, b & s, a & b.
    # If we store chosen as list of ints, it's fast.
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s margin
    
    trial = 0
    while time.time() - start_time < max_time:
        order = list(range(n_sets))
        rng.shuffle(order)
        
        chosen = []
        chosen_ints = []
        
        for idx in order:
            s = all_sets[idx]
            conflict_found = False
            
            # Check against all pairs in chosen
            # Optimization: if len(chosen) is small, this is fast.
            for i in range(len(chosen_ints)):
                a = chosen_ints[i]
                as_ = a & s
                # Early exit if as_ is not a subset of a? No, as_ is always subset.
                for j in range(i + 1, len(chosen_ints)):
                    b = chosen_ints[j]
                    ab = a & b
                    bs = b & s
                    if ab == as_ == bs:
                        conflict_found = True
                        break
                if conflict_found:
                    break
            
            if not conflict_found:
                chosen.append(idx)
                chosen_ints.append(s)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
            
        trial += 1
        # If we've done a few trials and time is up, break
        if time.time() - start_time > max_time:
            break
            
    return [sorted([i+1 for i in range(16) if (all_sets[idx] >> i) & 1]) for idx in best_chosen]
# EVOLVE-BLOCK-END
