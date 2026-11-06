import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 90s kill (evaluator kills at 90s, but we want to return well before)
    
    max_val = 4096
    max_sum = 3 * max_val + 1  # 12289
    # used_sums[i] is True if sum i is already taken
    used_sums = np.zeros(max_sum, dtype=bool)
    
    selected = []
    S = np.array([], dtype=np.int32)
    
    # We will try to build a set greedily.
    # To improve on the simple sequential greedy, we can use randomized restarts
    # or a smarter ordering.
    
    # The sequential greedy (1, 2, 3, ...) is fast and gets a decent set.
    # Let's run that first to establish a baseline.
    
    def run_greedy_sequential():
        nonlocal selected, S, used_sums
        selected = []
        S = np.array([], dtype=np.int32)
        used_sums[:] = False
        
        for x in range(1, max_val + 1):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if not selected:
                selected.append(x)
                S = np.array([x], dtype=np.int32)
                used_sums[3*x] = True
                continue
            
            k = len(selected)
            # Generate new triple sums involving x
            # Type 1: x + x + x
            s1 = 3 * x
            
            # Type 2: x + x + s_i
            s2 = 2 * x + S
            
            # Type 3: x + s_i + s_j with i <= j
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            
            all_new_sums = np.concatenate(([s1], s2, s3))
            
            # Check for duplicates within all_new_sums
            # and check against used_sums
            unique_new = np.unique(all_new_sums)
            if len(unique_new) != len(all_new_sums):
                continue
                
            if np.any(used_sums[all_new_sums]):
                continue
            
            # If valid, update
            selected.append(x)
            S = np.append(S, x)
            used_sums[all_new_sums] = True
            
        return selected
    
    best_set = run_greedy_sequential()
    best_size = len(best_set)
    
    # Now, let's try some randomized restarts to see if we can do better.
    # The idea is that the order of addition matters.
    # We can shuffle the candidates or pick them in a random order.
    
    def run_greedy_randomized(order):
        nonlocal selected, S, used_sums
        selected = []
        S = np.array([], dtype=np.int32)
        used_sums[:] = False
        
        for x in order:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if not selected:
                selected.append(x)
                S = np.array([x], dtype=np.int32)
                used_sums[3*x] = True
                continue
            
            k = len(selected)
            s1 = 3 * x
            s2 = 2 * x + S
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            
            all_new_sums = np.concatenate(([s1], s2, s3))
            
            unique_new = np.unique(all_new_sums)
            if len(unique_new) != len(all_new_sums):
                continue
                
            if np.any(used_sums[all_new_sums]):
                continue
            
            selected.append(x)
            S = np.append(S, x)
            used_sums[all_new_sums] = True
            
        return selected
    
    # Try a few randomized orders
    # We can also try biased orders, e.g., larger numbers first, or smaller numbers first.
    # The sequential greedy already tries smaller first.
    # Let's try larger first.
    
    order_desc = list(range(max_val, 0, -1))
    res_desc = run_greedy_randomized(order_desc)
    if len(res_desc) > best_size:
        best_set = res_desc
        best_size = len(res_desc)
    
    # Random shuffles
    # We need to be careful with the time budget.
    # Each run takes O(N * K^2) where N is the number of candidates and K is the size of the set.
    # N=4096, K~40. 4096 * 1600 = ~6.5M operations.
    # In NumPy, this is very fast.
    # Let's estimate how many runs we can do.
    
    while time.time() - start_time < DEADLINE - 5:
        # Create a random order
        # To make it more effective, we can mix strategies.
        # E.g., start with a random subset, then fill in gaps?
        # Or just shuffle the whole range.
        
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        res_rand = run_greedy_randomized(candidates)
        if len(res_rand) > best_size:
            best_set = res_rand
            best_size = len(res_rand)
            
    return best_set
