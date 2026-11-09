import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # 1. Generate 2,3,5-smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    
    # 2. Precompute conflict bitsets
    # conflict[i] is a bitmask where bit j is set if smooth[i] and smooth[j] are in conflict (one divides the other)
    # We also need to check if they are distinct. The problem says "distinct terms".
    # If i == j, they are the same term, so they are not distinct. But we pick distinct indices.
    # The condition is "no chosen term may divide another".
    # If a divides b and a != b, they conflict.
    
    conflict = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            # Check if smooth[i] divides smooth[j] or smooth[j] divides smooth[i]
            if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                conflict[i] |= (1 << j)
    
    # 3. Solve for each target 1..128
    solutions = [None] * 129
    
    # Helper: Check if adding index `idx` to `current_mask` is valid (antichain)
    # current_mask is a bitmask of indices already chosen.
    # If any chosen index conflicts with idx, it's invalid.
    # conflict[idx] has bits set for all indices that conflict with idx.
    # If (conflict[idx] & current_mask) != 0, then there is a conflict.
    
    def find_minimal_antichain(target):
        # Try k = 1, 2, ...
        for k in range(1, 31):
            if time.time() > deadline:
                return None
            
            # DFS to find a subset of size k summing to target
            # State: (start_idx, remaining_sum, current_mask, current_sum_list)
            # We can optimize by precomputing suffix sums of sorted smooth numbers?
            # Smooth numbers are sorted by value.
            # Let's just use standard DFS with pruning.
            
            # To prune effectively, we can sort candidates by value?
            # The smooth list is already sorted by value (1, 2, 3, 4, 5, 6, 8, 9, 10...).
            
            # We need to pick k items from smooth.
            # Pruning:
            # 1. If remaining_sum < 0, fail.
            # 2. If remaining items needed > available items from start_idx to end, fail.
            # 3. If sum of smallest 'remaining_items' items from start_idx > remaining_sum, fail? No, we need exactly remaining_sum.
            #    If sum of smallest 'remaining_items' items > remaining_sum, fail.
            #    If sum of largest 'remaining_items' items < remaining_sum, fail.
            
            # Precompute prefix sums? Or just calculate on the fly.
            # Since n_smooth is small (~30), calculating sum is fast.
            
            def dfs(start_idx, remaining_sum, remaining_k, current_mask, path):
                if time.time() > deadline:
                    return None
                
                if remaining_k == 0:
                    if remaining_sum == 0:
                        return path[:]
                    else:
                        return None
                
                if start_idx >= n_smooth:
                    return None
                
                # Pruning: Not enough elements left
                if n_smooth - start_idx < remaining_k:
                    return None
                
                # Pruning: Sum bounds
                # Min possible sum: sum of smallest remaining_k elements from start_idx
                # Max possible sum: sum of largest remaining_k elements from start_idx
                # But we don't have them sorted in a way that allows easy "smallest k from start_idx"
                # because we are iterating in order.
                # However, smooth is sorted.
                # So smallest remaining_k from start_idx are smooth[start_idx ... start_idx + remaining_k - 1]
                # Largest remaining_k from start_idx are smooth[n_smooth - remaining_k ... n_smooth - 1]
                # Wait, we can only pick from start_idx onwards.
                # So min_sum = sum(smooth[start_idx : start_idx + remaining_k])
                # max_sum = sum(smooth[n_smooth - remaining_k : n_smooth])
                # This max_sum bound is weak if we can't pick the largest ones because of index constraint?
                # Actually, we can pick any superset of indices >= start_idx.
                # So the largest possible sum is indeed picking the largest remaining_k from the entire list of available indices.
                # But we are constrained to pick from [start_idx, n_smooth).
                # So max_sum = sum(smooth[n_smooth - remaining_k : n_smooth]) is only valid if n_smooth - remaining_k >= start_idx.
                # If not, we take the last remaining_k available.
                
                min_sum = 0
                for i in range(start_idx, min(n_smooth, start_idx + remaining_k)):
                    min_sum += smooth[i]
                
                if min_sum > remaining_sum:
                    return None
                
                # Calculate max_sum more carefully
                # We need to pick remaining_k items from indices [start_idx, n_smooth-1]
                # The largest sum is obtained by picking the largest values.
                # Since smooth is sorted, these are the last remaining_k items in the range.
                start_max = max(start_idx, n_smooth - remaining_k)
                max_sum = 0
                for i in range(start_max, n_smooth):
                    max_sum += smooth[i]
                
                if max_sum < remaining_sum:
                    return None
                
                # Iterate through candidates
                # We can skip indices that are too large?
                # If smooth[i] > remaining_sum, we can't pick it.
                # But we can skip it.
                
                # Optimization: If we pick index i, we must pick remaining_k - 1 from [i+1, n_smooth)
                
                for i in range(start_idx, n_smooth):
                    val = smooth[i]
                    
                    # Pruning: If val > remaining_sum, and smooth is sorted, all subsequent are larger.
                    # But we can skip this val and pick smaller ones? No, we are iterating in increasing order.
                    # If val > remaining_sum, we cannot pick it.
                    # And since smooth is sorted, no subsequent val can be picked either (they are larger).
                    # Wait, we can skip i and try i+1? No, if smooth[i] > remaining_sum, then smooth[i+1] >= smooth[i] > remaining_sum.
                    # So we can break.
                    if val > remaining_sum:
                        break
                    
                    # Check antichain constraint
                    # If current_mask has any bit that conflicts with i, skip.
                    if conflict[i] & current_mask:
                        continue
                    
                    # Pick i
                    new_mask = current_mask | (1 << i)
                    path.append(i)
                    
                    res = dfs(i + 1, remaining_sum - val, remaining_k - 1, new_mask, path)
                    if res:
                        return res
                    
                    path.pop()
                    
                    # Pruning: If we didn't pick i, can we still achieve the sum?
                    # We need to pick remaining_k items from [i+1, n_smooth).
                    # This is handled by the loop continuing to i+1.
                    
                return None
            
            res_indices = dfs(0, target, k, 0, [])
            if res_indices:
                return [smooth[i] for i in res_indices]
        
        return None

    for t in range(1, 129):
        if time.time() > deadline:
            # Fallback: if we run out of time, we must return something valid for unsolved targets.
            # But we should try to solve them quickly.
            # If we are here, we failed to find a minimal one in time?
            # Unlikely for small k.
            pass
        
        sol = find_minimal_antichain(t)
        if sol:
            solutions[t] = sol
        else:
            # Fallback for safety (should not happen)
            # Try to find ANY antichain, not necessarily minimal.
            # Or just return [t] if t is smooth and doesn't divide anything?
            # If t is smooth, [t] is an antichain of size 1.
            if t in smooth:
                solutions[t] = [t]
            else:
                # This case is problematic. But problem guarantees existence.
                # We can try a larger k or just pick a valid one from a precomputed list if we had one.
                # For now, leave as None which will cause failure.
                # But let's try a quick non-minimal search if minimal failed?
                # Actually, if minimal failed, it's likely due to time or logic error.
                # Let's assume the DFS is correct and fast enough.
                pass

    return solutions[1:129]
