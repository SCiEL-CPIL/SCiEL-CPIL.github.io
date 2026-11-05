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
    compat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            a = smooth[i]
            b = smooth[j]
            if a % b != 0 and b % a != 0:
                compat[i][j] = True

    answers = []
    
    for target in range(1, 129):
        # Find candidates <= target
        start_idx = 0
        while start_idx < n_smooth and smooth[start_idx] > target:
            start_idx += 1
        
        candidates = smooth[start_idx:]
        cand_indices = list(range(start_idx, n_smooth))
        n_cand = len(candidates)
        
        if n_cand == 0:
            answers.append([target])
            continue

        # Precompute prefix sums of candidates for O(1) range sum queries
        prefix_sum = [0] * (n_cand + 1)
        for i in range(n_cand):
            prefix_sum[i+1] = prefix_sum[i] + candidates[i]
        
        def get_sum(l, r):
            if l > r: return 0
            if l < 0: l = 0
            if r >= n_cand: r = n_cand - 1
            return prefix_sum[r+1] - prefix_sum[l]

        found_sol = None
        
        for k in range(1, n_cand + 1):
            # Pruning: Min sum of k smallest candidates
            min_possible = get_sum(n_cand - k, n_cand - 1)
            if min_possible > target:
                break
            
            # Pruning: Max sum of k largest candidates
            max_possible = get_sum(0, k - 1)
            if max_possible < target:
                continue

            def dfs(pos, current_sum, current_global_indices, k_rem):
                if k_rem == 0:
                    if current_sum == target:
                        return current_global_indices[:]
                    return None
                
                if pos >= n_cand:
                    return None
                
                # Pruning 1: Max sum achievable with k_rem items from pos..end
                if pos + k_rem > n_cand:
                    return None
                max_add = get_sum(pos, pos + k_rem - 1)
                if current_sum + max_add < target:
                    return None
                
                # Pruning 2: Min sum achievable with k_rem items from pos..end
                # Smallest k_rem in candidates[pos..end] are the last k_rem of the whole list
                # provided they are >= pos.
                # Since we checked pos + k_rem <= n_cand, the last k_rem indices are valid.
                min_add = get_sum(n_cand - k_rem, n_cand - 1)
                if current_sum + min_add > target:
                    return None

                for i in range(pos, n_cand - k_rem + 1):
                    val = candidates[i]
                    global_idx = cand_indices[i]
                    
                    compatible = True
                    for prev_idx in current_global_indices:
                        if not compat[global_idx][prev_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    if current_sum + val > target:
                        continue
                        
                    res = dfs(i + 1, current_sum + val, current_global_indices + [global_idx], k_rem - 1)
                    if res is not None:
                        return res
                return None

            sol_indices = dfs(0, 0, [], k)
            if sol_indices is not None:
                found_sol = [smooth[idx] for idx in sol_indices]
                break
        
        if found_sol is None:
            found_sol = [target] if target in smooth else [1]
        
        answers.append(found_sol)
        
    return answers
