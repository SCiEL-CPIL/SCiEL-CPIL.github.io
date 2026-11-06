import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill

    max_val = 4096
    max_sum = 3 * max_val + 1
    # used_sums[i] is True if sum i is already taken
    used_sums = np.zeros(max_sum, dtype=bool)
    
    selected = []
    S = np.array([], dtype=np.int32)

    def is_valid_candidate(x, S, used_sums):
        """
        Check if x can be added to S.
        Returns True if valid, False otherwise.
        Also returns the new sums if valid (for optimization, but here we just check).
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
            # If k=1, triu_indices(1) gives (0,0), so s3 = x + S[0] + S[0] = x + 2*S[0]
            # This is covered by s2 if we are careful, but s2 is S+2x.
            # Wait.
            # Forms:
            # 1. x+x+x
            # 2. a+x+x for a in S
            # 3. a+b+x for a<=b in S
            # If k=1, S=[a].
            # s1: 3x
            # s2: a+2x
            # s3: a+a+x = 2a+x
            # My code above:
            # s2 = S + 2*x -> [a+2x]
            # s3: i_idx, j_idx = np.triu_indices(1) -> [0], [0]. s3 = x + S[0] + S[0] = x + 2a.
            # So s3 is [2a+x].
            # This is correct.
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            
        all_new = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        # If there are duplicates, x creates a collision with itself
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
        # Identify indices to remove from used_sums
        s1 = np.array([3 * x], dtype=np.int32)
        s2 = S + 2 * x
        # Note: S still contains x. We need to be careful.
        # The sums involving x that we added were:
        # 1. 3x
        # 2. s_i + 2x for all s_i in S (including x? No, s_i is from S before adding? 
        #    In add_element, S was the OLD set. So s2 used OLD S.
        #    So the sums to clear are based on OLD S.
        #    But here we have CURRENT S which includes x.
        #    Let's reconstruct the sums to clear based on S_without_x.
        
        S_old = np.delete(S, np.where(S == x)[0])
        
        s1_clear = np.array([3 * x], dtype=np.int32)
        s2_clear = S_old + 2 * x
        if len(S_old) > 0:
             i_idx, j_idx = np.triu_indices(len(S_old))
             s3_clear = x + S_old[i_idx] + S_old[j_idx]
        else:
             s3_clear = np.array([], dtype=np.int32)
             
        all_clear = np.concatenate((s1_clear, s2_clear, s3_clear))
        
        # Set used_sums to False for these sums
        # But only if no OTHER element in S_old creates these sums?
        # NO. The problem is that if another element y in S_old creates the same sum, 
        # then clearing it would be wrong.
        # However, the set S is VALID. This means all sums are unique.
        # Therefore, any sum created by x is NOT created by any other element in S.
        # So we can safely clear them.
        
        used_sums[all_clear] = False
        return S_old

    # Phase 1: Greedy Initial Construction
    # Try a few random orderings to find a good starting set
    best_set = []
    best_len = 0
    
    # Deterministic Greedy
    used_sums_init = np.zeros(max_sum, dtype=bool)
    S_init = np.array([], dtype=np.int32)
    for x in range(1, max_val + 1):
        if is_valid_candidate(x, S_init, used_sums_init):
            S_init = add_element(x, S_init, used_sums_init)
    
    best_set = S_init.tolist()
    best_len = len(best_set)

    # Phase 2: Local Search (Hill Climbing)
    # We try to improve the set by adding or swapping elements.
    # We work with a copy of the best set found.
    
    # We will run multiple local search iterations.
    # Time budget: ~40 seconds.
    
    # Helper to check if a set is valid (for debugging/safety, but we trust the incremental updates)
    # We maintain used_sums and S consistently.
    
    # Let's pick the best initial set and run local search on it.
    # We might also run local search on sets from random restarts.
    
    # To maximize the chance of finding size 40, we run local search from multiple random starts.
    
    num_restarts = 10
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
        
        # Pre-compute list of candidates not in S for faster sampling?
        # S is small (~40). Candidates are 4096.
        
        # We will iterate for a fixed number of steps or time.
        steps_per_restart = 2000
        for step in range(steps_per_restart):
            if time.time() - start_time > DEADLINE:
                break
                
            current_len = len(S_ls)
            
            # Strategy: 
            # 1. Try to Add.
            # Pick a random x from 1..4096 not in S_ls.
            # If valid, add it.
            
            # To pick x not in S_ls efficiently:
            # S_ls is small. We can pick random x and check if in S_ls.
            # Or maintain a list of availables.
            # Let's just pick random x and check membership.
            
            added = False
            swapped = False
            
            # Try Add
            # Bias towards smaller numbers? Or random?
            # Random is good for exploration.
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
                            # Note: add_element updates used_sums. 
                            # Since we reverted to S_temp (which had u removed and sums cleared),
                            # adding u back restores the state.
                            # Wait, remove_element cleared sums for u.
                            # add_element(u, S_temp) will mark them again.
                            # This is correct.
                            
            if len(S_ls) > best_len:
                best_len = len(S_ls)
                best_set = S_ls.tolist()
                # If we reached the theoretical max, we can stop early?
                # Max is 40.
                if best_len >= 40:
                    return best_set

    return best_set
