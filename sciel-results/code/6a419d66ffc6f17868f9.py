import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s kill

    max_val = 4096
    max_sum = 3 * max_val + 1

    # Precompute triu indices for k=0..100 to avoid repeated generation
    # This is a significant optimization for the inner loop
    triu_cache = {}
    for k in range(101):
        if k <= 1:
            triu_cache[k] = (np.array([0]*k, dtype=np.int32), np.array([0]*k, dtype=np.int32))
        else:
            triu_cache[k] = np.triu_indices(k)

    def is_valid_candidate(x, S, used_sums):
        """
        Check if x can be added to S.
        """
        k = len(S)
        if k == 0:
            return True
        
        # Generate all new sums involving x
        # 1. (x, x, x)
        s1 = np.array([3 * x], dtype=np.int32)
        # 2. (s_i, x, x)
        s2 = S + 2 * x
        
        # 3. (s_i, s_j, x) for i <= j
        i_idx, j_idx = triu_cache[k]
        s3 = x + S[i_idx] + S[j_idx]
        
        all_new = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        # Optimization: sort and check for adjacent duplicates is often faster than unique for small arrays
        # But np.unique is fine. Let's stick to unique for simplicity unless profiling says otherwise.
        if len(np.unique(all_new)) != len(all_new):
            return False
            
        # Check against used sums
        if np.any(used_sums[all_new]):
            return False
            
        return True

    def add_element(x, S, used_sums):
        """
        Add x to S and update used_sums.
        Assumes x is valid.
        """
        k = len(S)
        if k == 0:
            S_new = np.array([x], dtype=np.int32)
            used_sums[3 * x] = True
            return S_new
        
        s1 = np.array([3 * x], dtype=np.int32)
        s2 = S + 2 * x
        i_idx, j_idx = triu_cache[k]
        s3 = x + S[i_idx] + S[j_idx]
        all_new = np.concatenate((s1, s2, s3))
        
        used_sums[all_new] = True
        S_new = np.append(S, x)
        return S_new

    def remove_element(x, S, used_sums):
        """
        Remove x from S and update used_sums.
        Returns new S.
        """
        # Find index of x
        idx = np.where(S == x)[0]
        if len(idx) == 0:
            return S
        idx = idx[0]
        
        S_old = np.delete(S, idx)
        
        s1_clear = np.array([3 * x], dtype=np.int32)
        
        # s2_clear: sums of form (s_i, x, x) where s_i in S_old
        s2_clear = S_old + 2 * x
        
        # s3_clear: sums of form (s_i, s_j, x) where s_i, s_j in S_old, i<=j
        k_old = len(S_old)
        if k_old > 0:
            i_idx, j_idx = triu_cache[k_old]
            s3_clear = x + S_old[i_idx] + S_old[j_idx]
        else:
            s3_clear = np.array([], dtype=np.int32)
            
        all_clear = np.concatenate((s1_clear, s2_clear, s3_clear))
        
        # Clear only if not used by others? 
        # Since S is valid, sums are unique. So clearing is safe.
        used_sums[all_clear] = False
        return S_old

    # Phase 1: Deterministic Greedy
    used_sums_init = np.zeros(max_sum, dtype=bool)
    S_init = np.array([], dtype=np.int32)
    for x in range(1, max_val + 1):
        if is_valid_candidate(x, S_init, used_sums_init):
            S_init = add_element(x, S_init, used_sums_init)
    
    best_set = S_init.tolist()
    best_len = len(best_set)

    # Phase 2: Local Search with Random Restarts
    # Strategy:
    # 1. Start with a set (either deterministic greedy, or random greedy).
    # 2. Try to add elements.
    # 3. Try to swap elements to improve "density" or allow additions.
    # 4. Use a "soft" acceptance or simulated annealing? 
    #    Given the hard constraint, we can only accept valid moves.
    #    The objective is just to maximize size.
    #    So any valid add is an improvement.
    #    Swaps are neutral in size, but might enable future adds.
    
    num_restarts = 30  # Increase restarts since we have precomputed indices
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        # Random Restart Greedy
        random.seed(r + 123)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        used_sums_ls = np.zeros(max_sum, dtype=bool)
        S_ls = np.array([], dtype=np.int32)
        
        # Greedy on shuffled candidates
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if is_valid_candidate(x, S_ls, used_sums_ls):
                S_ls = add_element(x, S_ls, used_sums_ls)
                
        # Local Search Phase
        # Goal: Maximize len(S_ls)
        # Moves:
        # 1. Add a random valid x not in S.
        # 2. Swap u in S with v not in S if valid.
        
        # We will iterate for a fixed number of steps or time.
        steps_per_restart = 10000  # Increase steps due to speedup
        for step in range(steps_per_restart):
            if time.time() - start_time > DEADLINE:
                break
                
            current_len = len(S_ls)
            
            added = False
            swapped = False
            
            # Try Add
            # Pick a random x from 1..4096 not in S_ls.
            # To avoid picking from S_ls too often, we can maintain a list of availables?
            # S_ls is small. Random pick and check is fine.
            
            x_new = random.randint(1, max_val)
            if x_new not in S_ls:
                if is_valid_candidate(x_new, S_ls, used_sums_ls):
                    S_ls = add_element(x_new, S_ls, used_sums_ls)
                    added = True
                    
            if not added:
                # Try Swap
                # Pick u in S_ls, v not in S_ls
                if len(S_ls) > 0:
                    u = random.choice(S_ls.tolist())
                    v = random.randint(1, max_val)
                    if v != u and v not in S_ls:
                        # Remove u
                        S_temp = remove_element(u, S_ls, used_sums_ls)
                        # Check if v is valid with S_temp
                        if is_valid_candidate(v, S_temp, used_sums_ls):
                            # Accept swap
                            S_ls = add_element(v, S_temp, used_sums_ls)
                            swapped = True
                        else:
                            # Revert: Add u back to S_temp
                            S_ls = add_element(u, S_temp, used_sums_ls)
                            
            if len(S_ls) > best_len:
                best_len = len(S_ls)
                best_set = S_ls.tolist()
                # If we reached the theoretical max (40), we can stop early?
                # Max is 40.
                if best_len >= 40:
                    return best_set

    return best_set
