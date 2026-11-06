import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_val = 4096
    max_sum = 3 * max_val + 1
    used_sums = np.zeros(max_sum, dtype=bool)
    
    selected = []
    S = np.array([], dtype=np.int32)
    
    def try_add(x):
        nonlocal S, used_sums, selected
        
        if len(selected) == 0:
            selected.append(x)
            S = np.array([x], dtype=np.int32)
            used_sums[3*x] = True
            return True
        
        k = len(selected)
        
        # Generate new triple sums involving x
        # 1. (x, x, x)
        s1 = np.array([3*x], dtype=np.int32)
        
        # 2. (x, x, s) for s in S
        s2 = 2*x + S
        
        # 3. (x, s_i, s_j) for i <= j in S
        # We need all pairs (i, j) with i <= j
        i_idx, j_idx = np.triu_indices(k)
        s3 = x + S[i_idx] + S[j_idx]
        
        # Combine all new sums
        all_new_sums = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        # np.unique is O(n log n) but n is small (~3200 for k=80)
        # Alternatively, we can sort and check adjacent, but unique is fine.
        if len(np.unique(all_new_sums)) != len(all_new_sums):
            return False
            
        # Check against used_sums
        # all_new_sums contains indices into used_sums
        if np.any(used_sums[all_new_sums]):
            return False
            
        # If valid, update
        selected.append(x)
        S = np.append(S, x)
        used_sums[all_new_sums] = True
        return True

    # Primary strategy: Greedy from 1
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
        try_add(x)
    
    best_set = list(selected)
    best_len = len(best_set)
    
    # If time remains, try a few randomized restarts to see if we can beat the greedy result.
    # However, greedy is usually very strong for this type of problem.
    # We will only do this if the first greedy finished very quickly.
    elapsed = time.time() - start_time
    if elapsed < 10.0:
        import random
        random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Reset
            selected = []
            S = np.array([], dtype=np.int32)
            used_sums = np.zeros(max_sum, dtype=bool)
            
            # Randomize the first few choices slightly? 
            # Or just shuffle a prefix?
            # Let's try picking the first 3 elements randomly from a small range, then greedy.
            # This might help escape a bad local optimum if one exists.
            
            # Actually, a simpler randomization: skip some small numbers.
            # Greedy from 1 is optimal for many additive bases. 
            # Let's just run greedy again with a slight modification: 
            # Start from a random small offset? No, that likely reduces density.
            
            # Let's try: Greedy, but randomly skip 1 out of every 10 candidates in the first 100?
            # This is complex to implement efficiently with the vectorized check.
            
            # Given the high performance of the vectorized greedy, 
            # and the fact that previous scores were tied at 1.7778 (which corresponds to a specific size),
            # it's likely the greedy size is the bottleneck or the limit.
            # Let's just ensure the greedy is as fast as possible.
            
            # If we have time, we can try to "repair" or "improve" the set, but that's complex.
            # Let's just stick to the fast greedy.
            pass
            
    return best_set
