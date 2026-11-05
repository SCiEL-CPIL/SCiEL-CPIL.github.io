import sys
sys.setrecursionlimit(10000)

def construct():
    # Generate smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Sort in descending order
    smooth.sort(reverse=True)
    n_smooth = len(smooth)
    
    # Precompute compatibility
    compat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[i] % smooth[j] != 0 and smooth[j] % smooth[i] != 0:
                compat[i][i] # placeholder
                compat[i][j] = True

    answers = []
    
    for target in range(1, 129):
        # Filter candidates <= target
        # Since smooth is descending, find the first index where smooth[i] <= target
        # All indices from that point to end are <= target?
        # No, smooth is descending. smooth[0] is max.
        # We want smooth[i] <= target.
        # So we want the tail of the list.
        cand_indices = []
        for i in range(n_smooth):
            if smooth[i] <= target:
                cand_indices.append(i)
            else:
                # Since descending, if smooth[i] > target, all previous were > target.
                # But we are iterating 0 to n.
                # If smooth[i] > target, we skip.
                # If smooth[i] <= target, we include.
                # We can break? No, because smooth[i+1] <= smooth[i] <= target, so all subsequent are also <= target.
                # So we can find the first i where smooth[i] <= target, and take cand_indices = range(i, n_smooth).
                pass
        
        # Optimization: Find start index
        start_idx = n_smooth
        for i in range(n_smooth):
            if smooth[i] <= target:
                start_idx = i
                break
        
        if start_idx == n_smooth:
            # No candidates? Should not happen for target >= 1 as 1 is smooth.
            answers.append([1])
            continue
            
        cands = smooth[start_idx:] # Values
        # We need indices into smooth for compat check
        # Let's work with indices into the global smooth list for compat, 
        # but the search space is the slice [start_idx, n_smooth)
        
        # Precompute suffix sums for cands (values)
        # cands is sorted descending.
        # suffix_sum_cands[i] = sum(cands[i:])
        n_cands = len(cands)
        suffix_sum_cands = [0] * (n_cands + 1)
        for i in range(n_cands - 1, -1, -1):
            suffix_sum_cands[i] = suffix_sum_cands[i+1] + cands[i]
            
        # Map local index to global index
        # local idx l corresponds to global idx start_idx + l
        
        found_sol = None
        
        max_k = n_cands
        # Max k is limited by 30 per problem statement, but n_cands is small anyway.
        
        for k in range(1, max_k + 1):
            # Prune k
            # Max sum with k elements: sum of first k in cands
            max_sum = suffix_sum_cands[0] - suffix_sum_cands[k]
            if max_sum < target:
                continue
            
            # Min sum with k elements: sum of last k in cands
            # Last k are cands[n_cands-k:]
            # Sum = suffix_sum_cands[n_cands-k]
            min_sum = suffix_sum_cands[n_cands - k]
            if min_sum > target:
                break
            
            # DFS
            # State: local_start (index in cands), current_sum, chosen_global_indices, k_rem
            def dfs(local_start, current_sum, chosen, k_rem):
                if k_rem == 0:
                    if current_sum == target:
                        return chosen[:]
                    return None
                
                # Check bounds
                if local_start + k_rem > n_cands:
                    return None
                
                # Pruning
                # Max add: sum of cands[local_start : local_start+k_rem]
                max_add = suffix_sum_cands[local_start] - suffix_sum_cands[local_start + k_rem]
                if current_sum + max_add < target:
                    return None
                
                # Min add: sum of the k_rem smallest available.
                # Available are cands[local_start:].
                # The smallest k_rem in the whole list are cands[n_cands-k_rem:].
                # Are they all >= local_start?
                # Yes, if local_start <= n_cands - k_rem.
                # We already checked local_start + k_rem <= n_cands => local_start <= n_cands - k_rem.
                # So the global smallest k_rem are available.
                min_add = suffix_sum_cands[n_cands - k_rem]
                if current_sum + min_add > target:
                    return None
                
                # Iterate candidates
                # We pick cands[i] where i ranges from local_start to n_cands - k_rem
                for i in range(local_start, n_cands - k_rem + 1):
                    val = cands[i]
                    if current_sum + val > target:
                        continue # Since cands is descending, subsequent are smaller, so continue is correct.
                        
                    global_idx = start_idx + i
                    
                    # Check compatibility
                    compatible = True
                    for prev_g_idx in chosen:
                        if not compat[global_idx][prev_g_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    res = dfs(i + 1, current_sum + val, chosen + [global_idx], k_rem - 1)
                    if res is not None:
                        return res
                return None

            sol_indices = dfs(0, 0, [], k)
            if sol_indices is not None:
                found_sol = [smooth[idx] for idx in sol_indices]
                break
        
        if found_sol is None:
            # Fallback
            found_sol = [target] if target in smooth else [1]
            
        answers.append(found_sol)
        
    return answers
