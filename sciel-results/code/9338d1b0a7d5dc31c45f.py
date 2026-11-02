<<<<<<< SEARCH
import numpy as np
import itertools
import time
=======
import numpy as np
import itertools
import time
import random
>>>>>>> REPLACE
<<<<<<< SEARCH
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    
    # Try multiple random tie-breakings
    # The greedy algorithm is:
    # 1. active = all True
    # 2. deg = sum(Conflict, axis=1)
    # 3. While active:
    #    a. Find active nodes with min deg
    #    b. Pick one (randomly if tie, or first)
    #    c. Add to set
    #    d. Remove it and its neighbors
    #    e. Update degrees
    
    def run_greedy(seed):
=======
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Build bitsets for conflict graph
    conflict_bits = [0] * N
    for i in range(N):
        bits = 0
        row = Conflict[i, :]
        indices = np.where(row)[0]
        for j in indices:
            bits |= (1 << j)
        conflict_bits[i] = bits

    def run_greedy(seed):
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Helper for local search: try to improve a set
    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        # Precompute conflict mask for selected set? No, use global Conflict matrix.
        # We want to find i in selected, j,k not in selected
        # such that j,k compatible with each other
        # j compatible with selected \ {i}
        # k compatible with selected \ {i}
        
        # This is O(N^2) per iteration which is too slow for N=5000.
        # We will do a lighter version: 
        # For each i in selected, find all j not in selected that conflict ONLY with i.
        # If we find two such j1, j2 that are compatible with each other, swap i -> j1, j2.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffled order to break symmetry
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find candidates j not in selected_set that conflict with i
                # and do NOT conflict with any other element in selected_set
                # conflict_mask_i = Conflict[i]
                # selected_mask = np.zeros(N, dtype=bool)
                # selected_mask[list(selected_set)] = True
                
                # Efficient check:
                # Let S = selected_set.
                # For j not in S, let C_j = set of elements in S that conflict with j.
                # We want |C_j| == 1 and that element is i.
                
                # Compute conflicts of i with all N
                conflicts_with_i = Conflict[i].astype(bool)
                
                # For each j not in S:
                # Check if j conflicts with any k in S \ {i}
                # This is expensive.
                
                # Alternative:
                # We know j must conflict with i. So j is in neighbors of i.
                # Let neighbors_of_i = np.where(conflicts_with_i)[0]
                # Filter neighbors_of_i to those not in S.
                # For each candidate j in that list:
                #   Check if j conflicts with any k in S \ {i}.
                #   If not, j is a "valid replacement candidate" for i (single conflict).
                
                neighbors_of_i = np.where(conflicts_with_i)[0]
                # Exclude those already in S
                valid_js = [j for j in neighbors_of_i if j not in selected_set]
                
                # Now find pairs (j1, j2) in valid_js such that:
                # 1. j1 and j2 are compatible (Conflict[j1, j2] == 0)
                # 2. j1 does not conflict with any k in S \ {i}
                # 3. j2 does not conflict with any k in S \ {i}
                
                # Precompute "bad" sets for S \ {i}?
                # Let's just check on the fly for small lists.
                # If valid_js is large, this is slow.
                
                # Optimization:
                # A vector j is "compatible with S\{i}" if Conflict[j, S\{i}].sum() == 0.
                
                compatible_with_rest = []
                for j in valid_js:
                    # Check conflicts with S \ {i}
                    # S is a set. We can iterate.
                    # To speed up, if len(S) is large, this is bad.
                    # However, greedy sets are usually smallish relative to N.
                    conflict_count = 0
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict[j, k]:
                            conflict_count += 1
                            break
                    if conflict_count == 0:
                        compatible_with_rest.append(j)
                
                # Now look for a pair in compatible_with_rest that are mutually compatible
                if len(compatible_with_rest) >= 2:
                    # Check pairs
                    found_pair = False
                    cwr = compatible_with_rest
                    for p in range(len(cwr)):
                        if found_pair: break
                        j1 = cwr[p]
                        for q in range(p+1, len(cwr)):
                            j2 = cwr[q]
                            if not Conflict[j1, j2]:
                                # Swap i -> j1, j2
                                selected_set.remove(i)
                                selected_set.add(j1)
                                selected_set.add(j2)
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        # Restart the loop to re-evaluate
                        break
                
        return list(selected_set)
=======
    def local_search_bitset(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        S = 0
        for i in selected_indices:
            S |= (1 << i)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get list of selected indices for iteration
            s_list = []
            temp_S = S
            while temp_S:
                lsb = temp_S & -temp_S
                idx = lsb.bit_length() - 1
                s_list.append(idx)
                temp_S ^= lsb
                
            random.shuffle(s_list)
            
            for i in s_list:
                if time.time() > deadline:
                    res = []
                    temp_S = S
                    while temp_S:
                        lsb = temp_S & -temp_S
                        idx = lsb.bit_length() - 1
                        res.append(idx)
                        temp_S ^= lsb
                    return res
                    
                bit_i = 1 << i
                S_without_i = S & ~bit_i
                
                neighbors_i = conflict_bits[i]
                potential = neighbors_i & ~S
                
                if potential == 0:
                    continue
                    
                compatible_js = []
                p = potential
                while p:
                    lsb = p & -p
                    j = lsb.bit_length() - 1
                    p ^= lsb
                    
                    if (conflict_bits[j] & S_without_i) == 0:
                        compatible_js.append(j)
                        
                if len(compatible_js) >= 2:
                    found_pair = False
                    cwr = compatible_js
                    for a in range(len(cwr)):
                        if found_pair: break
                        j1 = cwr[a]
                        for b in range(a+1, len(cwr)):
                            j2 = cwr[b]
                            if not (conflict_bits[j1] >> j2) & 1:
                                S = S_without_i | (1 << j1) | (1 << j2)
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break
                        
        res = []
        temp_S = S
        while temp_S:
            lsb = temp_S & -temp_S
            idx = lsb.bit_length() - 1
            res.append(idx)
            temp_S ^= lsb
        return res
>>>>>>> REPLACE
<<<<<<< SEARCH
    for s in range(100):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply local search
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        # Fallback
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
=======
    for s in range(200):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply local search
        res_improved = local_search_bitset(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        # Fallback
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
>>>>>>> REPLACE
