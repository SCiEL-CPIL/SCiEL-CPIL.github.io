import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)
    smooth_set = set(smooth)

    # 2. Precompute conflict masks
    # conflict_mask[i] is a bitmask of indices j such that smooth[i] and smooth[j] conflict
    # Conflict means one divides the other.
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                    conflict_mask[i] |= (1 << j)

    # 3. Global Search: Find minimal antichain for each target 1..128
    # We will enumerate all valid antichains with sum <= 128.
    # best_size[t] = minimal number of terms
    # best_terms[t] = the list of terms
    
    best_size = [999] * 129
    best_terms = [None] * 129
    
    # Initialize single-term antichains (size 1)
    for i in range(n_smooth):
        val = smooth[i]
        if val <= 128:
            if 1 < best_size[val]:
                best_size[val] = 1
                best_terms[val] = [val]

    # DFS to find larger antichains
    # We want to find minimal size, so we should prioritize finding small sets.
    # However, standard DFS finds all. We can prune if current_size >= best_size[current_sum] is not directly possible 
    # because current_sum changes.
    # Instead, we just generate all and keep the min.
    # To ensure we don't blow up, we rely on the sum constraint.
    
    def dfs(start_idx, current_sum, current_mask, current_terms, current_size):
        if time.time() > deadline:
            return
        
        # If current sum is a valid target, update if size is smaller
        if current_sum <= 128:
            if current_size < best_size[current_sum]:
                best_size[current_sum] = current_size
                best_terms[current_sum] = list(current_terms)
        
        # Pruning: if we add any more elements, sum will increase.
        # We can try to add elements from start_idx onwards.
        for i in range(start_idx, n_smooth):
            val = smooth[i]
            new_sum = current_sum + val
            if new_sum > 128:
                break # Since smooth is sorted, no larger val will fit
            
            # Check conflict with current set
            if current_mask & conflict_mask[i]:
                continue
            
            new_mask = current_mask | (1 << i)
            current_terms.append(val)
            dfs(i + 1, new_sum, new_mask, current_terms, current_size + 1)
            current_terms.pop()

    # Start DFS
    # We can start with empty set
    dfs(0, 0, 0, [], 0)

    # 4. Handle any missing targets (should not happen, but safe)
    # If a target has no representation, it means no antichain sums to it.
    # But the problem states "All targets require valid representations".
    # If best_size[n] is still 999, we need a fallback.
    # The fallback is the original per-target search, but only for missing ones.
    
    missing = [n for n in range(1, 129) if best_size[n] == 999]
    
    if missing:
        # Fallback for missing targets: use the old DFS method
        # This is a safety net.
        
        def find_min_antichain_fallback(target):
            for k in range(1, 31):
                if time.time() > deadline:
                    return None
                if k == 1:
                    if target in smooth_set:
                        return [target]
                    continue
                # Simple DFS for this specific target
                memo = {}
                def dfs_fb(remaining, k_left, start_idx, mask):
                    if remaining < 0 or k_left < 0:
                        return None
                    if k_left == 0:
                        return () if remaining == 0 else None
                    if start_idx + k_left > n_smooth:
                        return None
                    # Pruning: min sum of k_left smallest from start_idx
                    min_sum = sum(smooth[start_idx:start_idx+k_left])
                    if min_sum > remaining:
                        return None
                    key = (remaining, k_left, start_idx, mask)
                    if key in memo:
                        return memo[key]
                    for i in range(start_idx, n_smooth):
                        val = smooth[i]
                        if val > remaining:
                            break
                        if mask & conflict_mask[i]:
                            continue
                        new_mask = mask | (1 << i)
                        res = dfs_fb(remaining - val, k_left - 1, i + 1, new_mask)
                        if res is not None:
                            memo[key] = (i,) + res
                            return (i,) + res
                    memo[key] = None
                    return None
                res = dfs_fb(target, k, 0, 0)
                if res is not None:
                    return [smooth[i] for i in res]
            return None

        for n in missing:
            res = find_min_antichain_fallback(n)
            if res:
                best_terms[n] = res
                best_size[n] = len(res)
            else:
                # Last resort: if n is smooth, use [n]
                if n in smooth_set:
                    best_terms[n] = [n]
                    best_size[n] = 1
                else:
                    # This should not happen
                    best_terms[n] = []
                    best_size[n] = 0

    # 5. Construct output
    answers = []
    for n in range(1, 129):
        if best_terms[n] is None:
            # Should not happen
            answers.append([])
        else:
            answers.append(best_terms[n])
            
    return answers
