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
    
    # Sort in descending order to help pruning (larger numbers first)
    smooth.sort(reverse=True)
    n_smooth = len(smooth)
    
    # Precompute compatibility: compat[i][j] is True if smooth[i] and smooth[j] are incomparable
    # i.e., smooth[i] % smooth[j] != 0 and smooth[j] % smooth[i] != 0
    compat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            a = smooth[i]
            b = smooth[j]
            if a % b != 0 and b % a != 0:
                compat[i][j] = True
            else:
                compat[i][j] = False

    # Precompute suffix sums for pruning
    # suffix_sum[i][k] = sum of largest k elements in smooth[i:]
    # This is an upper bound on the sum we can add with k more elements
    suffix_sum = [[0] * (n_smooth + 1) for _ in range(n_smooth + 1)]
    for i in range(n_smooth - 1, -1, -1):
        for k in range(1, n_smooth - i + 1):
            # The largest k elements in smooth[i:] are just smooth[i], smooth[i+1], ... smooth[i+k-1]
            # Because smooth is sorted descending.
            suffix_sum[i][k] = suffix_sum[i+1][k-1] + smooth[i]

    answers = []
    
    for target in range(1, 129):
        # Find candidates <= target
        # Since smooth is sorted descending, we can find the index where smooth[i] <= target
        # All smooth[i] for i >= start_idx are <= target? No, smooth[0] is largest.
        # We want elements <= target.
        # Let's filter.
        candidates = []
        for idx, val in enumerate(smooth):
            if val <= target:
                candidates.append(idx)
            else:
                break # Since sorted descending, once val > target, all subsequent are smaller? 
                # Wait. smooth is descending. smooth[0] >= smooth[1] ...
                # If smooth[0] > target, we skip.
                # If smooth[i] <= target, then smooth[i+1] <= smooth[i] <= target.
                # So we can break when we find the first one <= target? No.
                # We want ALL <= target.
                # Since descending, the ones <= target are at the END of the list.
                # Example: [128, 125, 120, ..., 1]
                # Target 50.
                # 128 > 50
                # 125 > 50
                # ...
                # 50 <= 50 -> include
                # 48 <= 50 -> include
                # So we should iterate and collect those <= target.
                # Or, since it's sorted, we can find the split point.
                pass
        
        # Re-do candidate selection properly
        cand_indices = []
        for i in range(n_smooth):
            if smooth[i] <= target:
                cand_indices.append(i)
        
        found = None
        max_k = len(cand_indices)
        
        for k in range(1, max_k + 1):
            # Try to find a subset of size k from cand_indices
            # that sums to target and is an antichain.
            
            # Pruning: Max possible sum with k elements from cand_indices
            # The largest k elements in cand_indices are the first k in the list?
            # cand_indices preserves the order of smooth (descending).
            # So cand_indices[0] is the largest candidate.
            # Max sum = sum(smooth[cand_indices[i]] for i in range(k))
            max_sum = sum(smooth[idx] for idx in cand_indices[:k])
            if max_sum < target:
                continue # Even taking the largest k, we can't reach target
            
            # Min possible sum with k elements
            min_sum = sum(smooth[idx] for idx in cand_indices[-k:])
            if min_sum > target:
                break # Even taking the smallest k, we exceed target. 
                       # Since k is increasing, min_sum will decrease? 
                       # No, as k increases, we add smaller numbers, so min_sum decreases.
                       # Wait.
                       # k=1: min_sum = smallest candidate.
                       # k=2: min_sum = two smallest candidates.
                       # So if min_sum > target for some k, it will be even smaller for k+1?
                       # Yes. So if min_sum > target, we can continue to larger k?
                       # No, if the smallest k sum to > target, then any k will sum to > target?
                       # No. 
                       # Example: target 10. Candidates [9, 8, 1].
                       # k=1: min=1, max=9.
                       # k=2: min=1+8=9, max=9+8=17.
                       # k=3: min=1+8+9=18 > 10.
                       # So if min_sum > target, we can break?
                       # Yes, because adding more terms only increases the sum (terms are positive).
                       # So if the sum of the k SMALLEST terms is > target, then any subset of size k sums to > target.
                       # And any subset of size > k sums to even more.
                       # So we can break the k loop.
            if min_sum > target:
                break

            # DFS to find valid antichain of size k
            # We pick indices from cand_indices
            # To avoid permutations, we pick increasing indices from cand_indices
            # But we need to check compatibility.
            
            def dfs(start_idx, current_sum, current_set, k_rem):
                if k_rem == 0:
                    return current_sum == target
                # Pruning
                # 1. Current sum + max possible sum < target
                # Max possible sum from cand_indices[start_idx:] with k_rem elements
                # These are the largest k_rem elements available from start_idx onwards.
                # Since cand_indices is sorted descending (inherited from smooth),
                # the largest are at the beginning of the slice.
                if start_idx + k_rem > len(cand_indices):
                    return False
                max_add = sum(smooth[cand_indices[i]] for i in range(start_idx, start_idx + k_rem))
                if current_sum + max_add < target:
                    return False
                
                # 2. Current sum + min possible sum > target
                # Min possible sum is the sum of the k_rem SMALLEST elements available.
                # These are at the end of cand_indices.
                # But we are constrained to start_idx onwards.
                # So the smallest are cand_indices[-1], cand_indices[-2]...
                # However, we must pick from start_idx to end.
                # The smallest k_rem in cand_indices[start_idx:] are the last k_rem of that slice.
                slice_len = len(cand_indices) - start_idx
                if slice_len < k_rem:
                    return False
                min_add = sum(smooth[cand_indices[i]] for i in range(len(cand_indices) - k_rem, len(cand_indices)))
                # Wait, this min_add is for the WHOLE list.
                # We are restricted to start_idx onwards.
                # The smallest k_rem in cand_indices[start_idx:] are cand_indices[-k_rem:] if start_idx <= len-k_rem.
                # Yes.
                if current_sum + min_add > target:
                    return False

                for i in range(start_idx, len(cand_indices) - k_rem + 1):
                    idx = cand_indices[i]
                    val = smooth[idx]
                    
                    # Check if val is compatible with all in current_set
                    # current_set contains indices into smooth
                    compatible = True
                    for prev_idx in current_set:
                        if not compat[idx][prev_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    # Prune if current_sum + val > target
                    if current_sum + val > target:
                        continue # Since smooth is descending, subsequent vals are smaller, so we can't break?
                        # Wait. smooth is descending. cand_indices is in order of smooth.
                        # So cand_indices[i] >= cand_indices[i+1].
                        # If val > target - current_sum, then val is too big.
                        # Subsequent vals are smaller, so they might fit.
                        # So we cannot break, we must continue.
                        pass
                    
                    if dfs(i + 1, current_sum + val, current_set + [idx], k_rem - 1):
                        return True
                return False

            if dfs(0, 0, [], k):
                # Reconstruct the solution? 
                # The dfs returns True/False. We need the actual terms.
                # Let's modify dfs to return the list of indices or None.
                pass
        # If not found, something is wrong.
        # But the problem guarantees a solution.
        # Let's rewrite dfs to return the solution.
        
        # Re-do the k loop with a solution-returning DFS
        found_sol = None
        for k in range(1, max_k + 1):
            max_sum = sum(smooth[idx] for idx in cand_indices[:k])
            if max_sum < target:
                continue
            min_sum = sum(smooth[idx] for idx in cand_indices[-k:])
            if min_sum > target:
                break
            
            def dfs_sol(start_idx, current_sum, current_set, k_rem):
                if k_rem == 0:
                    if current_sum == target:
                        return current_set[:]
                    return None
                
                # Pruning
                if start_idx + k_rem > len(cand_indices):
                    return None
                max_add = sum(smooth[cand_indices[i]] for i in range(start_idx, start_idx + k_rem))
                if current_sum + max_add < target:
                    return None
                
                min_add = sum(smooth[cand_indices[i]] for i in range(len(cand_indices) - k_rem, len(cand_indices)))
                if current_sum + min_add > target:
                    return None

                for i in range(start_idx, len(cand_indices) - k_rem + 1):
                    idx = cand_indices[i]
                    val = smooth[idx]
                    
                    compatible = True
                    for prev_idx in current_set:
                        if not compat[idx][prev_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    if current_sum + val > target:
                        continue
                    
                    res = dfs_sol(i + 1, current_sum + val, current_set + [idx], k_rem - 1)
                    if res is not None:
                        return res
                return None

            sol_indices = dfs_sol(0, 0, [], k)
            if sol_indices is not None:
                found_sol = [smooth[idx] for idx in sol_indices]
                break
        
        if found_sol is None:
            # Fallback: should not happen
            found_sol = [target] if target in smooth else [1]
        
        answers.append(found_sol)
        
    return answers
