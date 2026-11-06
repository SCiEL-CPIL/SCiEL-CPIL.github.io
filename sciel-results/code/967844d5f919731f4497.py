import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill

    max_val = 4096
    max_sum = 3 * max_val + 1
    
    def get_new_sums(x, S):
        """
        Calculate all triple sums involving x and elements of S.
        Assumes x is not in S.
        """
        k = len(S)
        if k == 0:
            return np.array([3 * x], dtype=np.int32)
        
        s1 = np.array([3 * x], dtype=np.int32)
        s2 = S + 2 * x
        if k > 1:
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
        else:
            s3 = np.array([2 * S[0] + x], dtype=np.int32)
            
        return np.concatenate((s1, s2, s3))

    def is_valid_candidate(x, S, used_sums):
        """
        Check if x can be added to S.
        """
        # Check if x is already in S
        # S is a numpy array. 
        # For small S, linear search is fast.
        for val in S:
            if val == x:
                return False

        new_sums = get_new_sums(x, S)
        
        # Check for internal duplicates
        if len(np.unique(new_sums)) != len(new_sums):
            return False
            
        # Check against existing used sums
        if np.any(used_sums[new_sums]):
            return False
            
        return True

    def add_element(x, S, used_sums):
        """
        Add x to S and update used_sums.
        """
        new_sums = get_new_sums(x, S)
        used_sums[new_sums] = True
        S_new = np.append(S, x)
        return S_new

    def remove_element(x, S, used_sums):
        """
        Remove x from S and update used_sums.
        """
        S_old = np.delete(S, np.where(S == x)[0])
        
        if len(S_old) == 0:
            used_sums[3 * x] = False
            return S_old
            
        s1_clear = np.array([3 * x], dtype=np.int32)
        s2_clear = S_old + 2 * x
        if len(S_old) > 1:
            i_idx, j_idx = np.triu_indices(len(S_old))
            s3_clear = x + S_old[i_idx] + S_old[j_idx]
        else:
            s3_clear = np.array([2 * S_old[0] + x], dtype=np.int32)
            
        all_clear = np.concatenate((s1_clear, s2_clear, s3_clear))
        used_sums[all_clear] = False
        return S_old

    def greedy_construct(ordering):
        """
        Run greedy construction with a specific ordering of candidates.
        """
        used_sums_g = np.zeros(max_sum, dtype=bool)
        S_g = np.array([], dtype=np.int32)
        
        for x in ordering:
            if time.time() - start_time > DEADLINE:
                break
            if is_valid_candidate(x, S_g, used_sums_g):
                S_g = add_element(x, S_g, used_sums_g)
                
        return S_g

    # Phase 1: Multiple Greedy Starts
    best_set = np.array([], dtype=np.int32)
    best_len = 0
    
    # 1. Ascending order
    S_asc = greedy_construct(range(1, max_val + 1))
    if len(S_asc) > best_len:
        best_len = len(S_asc)
        best_set = S_asc.copy()
        
    # 2. Descending order
    S_desc = greedy_construct(range(max_val, 0, -1))
    if len(S_desc) > best_len:
        best_len = len(S_desc)
        best_set = S_desc.copy()
        
    # 3. Random orders
    for i in range(20):
        if time.time() - start_time > DEADLINE:
            break
        random.seed(i + 100)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        S_rand = greedy_construct(candidates)
        if len(S_rand) > best_len:
            best_len = len(S_rand)
            best_set = S_rand.copy()

    # Phase 2: Local Search
    # Reconstruct used_sums for best_set
    used_sums_ls = np.zeros(max_sum, dtype=bool)
    S_ls = np.array([], dtype=np.int32)
    for x in best_set:
        S_ls = add_element(x, S_ls, used_sums_ls)
    
    current_len = len(S_ls)
    no_improvement_count = 0
    max_no_improvement = 20000 # Increased to allow more exploration
    
    while time.time() - start_time < DEADLINE:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        
        # 1. Try to Add
        # Try a few random candidates to increase chance of finding a valid one
        add_success = False
        for _ in range(5):
            x_new = random.randint(1, max_val)
            if x_new not in S_ls:
                if is_valid_candidate(x_new, S_ls, used_sums_ls):
                    S_ls = add_element(x_new, S_ls, used_sums_ls)
                    current_len += 1
                    improved = True
                    no_improvement_count = 0
                    add_success = True
                    break
        
        if add_success:
            continue
            
        # 2. Try Single Swap
        swap_success = False
        if len(S_ls) > 0:
            u = random.choice(S_ls.tolist())
            for _ in range(10):
                v = random.randint(1, max_val)
                if v == u or v in S_ls:
                    continue
                    
                S_temp = remove_element(u, S_ls, used_sums_ls)
                
                if is_valid_candidate(v, S_temp, used_sums_ls):
                    S_ls = add_element(v, S_temp, used_sums_ls)
                    improved = True
                    no_improvement_count = 0
                    swap_success = True
                    break
                else:
                    S_ls = add_element(u, S_temp, used_sums_ls)
                    
        if swap_success:
            continue

        # 3. Try Double Swap
        double_swap_success = False
        if len(S_ls) >= 2:
            indices = np.random.choice(len(S_ls), 2, replace=False)
            u = S_ls[indices[0]]
            w = S_ls[indices[1]]
            
            S_temp1 = remove_element(u, S_ls, used_sums_ls)
            S_temp2 = remove_element(w, S_temp1, used_sums_ls)
            
            v = random.randint(1, max_val)
            z = random.randint(1, max_val)
            
            # Ensure v, z are not in S_temp2 and distinct
            if v != z and v not in S_temp2 and z not in S_temp2:
                if is_valid_candidate(v, S_temp2, used_sums_ls):
                    S_temp3 = add_element(v, S_temp2, used_sums_ls)
                    if is_valid_candidate(z, S_temp3, used_sums_ls):
                        S_ls = add_element(z, S_temp3, used_sums_ls)
                        improved = True
                        no_improvement_count = 0
                        double_swap_success = True
                    else:
                        # Revert v
                        S_temp2 = remove_element(v, S_temp3, used_sums_ls)
                        
            if not double_swap_success:
                # Revert w, u
                S_temp1_restored = add_element(w, S_temp2, used_sums_ls)
                S_ls = add_element(u, S_temp1_restored, used_sums_ls)
                
        if not improved:
            no_improvement_count += 1

    return S_ls.tolist()
