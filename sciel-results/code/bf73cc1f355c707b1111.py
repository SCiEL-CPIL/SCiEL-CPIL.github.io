import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill

    max_val = 4096
    max_sum = 3 * max_val + 1
    # used_sums[i] is True if sum i is already taken

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
        if k > 1:
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
        else:
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            
        all_new = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
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
        i_idx, j_idx = np.triu_indices(k)
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
        k = len(S)
        S_old = np.delete(S, np.where(S == x)[0])
        
        s1_clear = np.array([3 * x], dtype=np.int32)
        s2_clear = S_old + 2 * x
        if len(S_old) > 0:
             i_idx, j_idx = np.triu_indices(len(S_old))
             s3_clear = x + S_old[i_idx] + S_old[j_idx]
        else:
             s3_clear = np.array([], dtype=np.int32)
             
        all_clear = np.concatenate((s1_clear, s2_clear, s3_clear))
        
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
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        random.seed(r + 123)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        used_sums_ls = np.zeros(max_sum, dtype=bool)
        S_ls = np.array([], dtype=np.int32)
        
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if is_valid_candidate(x, S_ls, used_sums_ls):
                S_ls = add_element(x, S_ls, used_sums_ls)
                
        # Local Search Phase
        steps_per_restart = 8000
        for step in range(steps_per_restart):
            if time.time() - start_time > DEADLINE:
                break
                
            current_len = len(S_ls)
            
            added = False
            swapped = False
            
            # Try Add with multiple random attempts
            for _attempt in range(3):
                x_new = random.randint(1, max_val)
                if x_new not in S_ls:
                    if is_valid_candidate(x_new, S_ls, used_sums_ls):
                        S_ls = add_element(x_new, S_ls, used_sums_ls)
                        added = True
                        break
                    
            if not added:
                # Try Swap with multiple u,v combinations
                if len(S_ls) > 0:
                    u = random.choice(S_ls.tolist())
                    S_temp = remove_element(u, S_ls, used_sums_ls)
                    
                    swap_done = False
                    for _v_attempt in range(5):
                        v = random.randint(1, max_val)
                        if v != u and v not in S_temp:
                            if is_valid_candidate(v, S_temp, used_sums_ls):
                                S_ls = add_element(v, S_temp, used_sums_ls)
                                swapped = True
                                swap_done = True
                                break
                    
                    if not swap_done:
                        # Revert: Add u back to S_temp
                        S_ls = add_element(u, S_temp, used_sums_ls)
                            
            if len(S_ls) > best_len:
                best_len = len(S_ls)
                best_set = S_ls.tolist()
                if best_len >= 40:
                    return best_set

    return best_set
