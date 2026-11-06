import random
import time
import numpy as np
from itertools import combinations

def construct():
    n = 16
    k = 4
    # Generate all k-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    all_sets_np = np.array(all_sets, dtype=np.int32)
    
    best_family = []
    start_time = time.time()
    max_time = 55.0 # Leave buffer for overhead
    
    # Pre-shuffle indices to avoid modifying the original array
    # We will use random permutation of indices
    
    while time.time() - start_time < max_time:
        # Random permutation of all set indices
        perm = np.random.permutation(len(all_sets_np))
        shuffled_sets = all_sets_np[perm]
        
        family = []
        # Store pairs as lists for easy appending, convert to numpy for checking
        pair_a_list = []
        pair_b_list = []
        pair_i_list = []
        
        for s in shuffled_sets:
            if len(family) == 0:
                family.append(int(s))
                continue
            
            # Check for conflict using numpy
            # Convert lists to arrays for vectorized operation
            # Optimization: Only convert if we have a reasonable number of pairs
            # Since N is small, this is fast.
            if pair_a_list:
                pa = np.array(pair_a_list, dtype=np.int32)
                pb = np.array(pair_b_list, dtype=np.int32)
                pi = np.array(pair_i_list, dtype=np.int32)
                
                # Vectorized check: (pa & s == pi) AND (pb & s == pi)
                # s is a numpy scalar or int, need to ensure broadcasting works
                s_val = np.int32(s)
                cond = ((pa & s_val) == pi) & ((pb & s_val) == pi)
                conflict = np.any(cond)
            else:
                conflict = False
            
            if not conflict:
                s_int = int(s)
                family.append(s_int)
                # Update pairs
                for a in family[:-1]:
                    i = a & s_int
                    pair_a_list.append(a)
                    pair_b_list.append(s_int)
                    pair_i_list.append(i)
        
        if len(family) > len(best_family):
            best_family = family
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
