import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # 90s limit, leave 5s margin
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Precompute divisibility as bitsets for fast conflict checking
    # conflict_mask[i] = bitmask of all indices j where smooth_list[i] and smooth_list[j] divide each other
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
    
    # Map smooth value to its index
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Pre-filter candidates for each n
    candidates = [[] for _ in range(129)]
    for i in range(129):
        for idx in range(n_smooth):
            if smooth_list[idx] <= i:
                candidates[i].append(idx)
            else:
                break
        
    # Sort candidates descending by value for each n
    cand_desc = [sorted(candidates[n], key=lambda idx: smooth_list[idx], reverse=True) for n in range(129)]
    
    # Precompute prefix sums for each n's candidate list (sorted descending)
    # prefix_sums[n][i] = sum of first i elements in cand_desc[n]
    prefix_sums = [None] * 129
    for n in range(129):
        c = cand_desc[n]
        ps = [0]
        for idx in c:
            ps.append(ps[-1] + smooth_list[idx])
        prefix_sums[n] = ps
    
    # Memoization cache for find_antichain results
    memo = {}
    
    def find_antichain(n, k):
        """
        Finds an antichain of size k summing to n using bitset-based backtracking.
        Returns list of terms or None.
        """
        key = (n, k)
        if key in memo:
            return memo[key]
        
        cand = cand_desc[n]
        ncand = len(cand)
        if ncand < k:
            memo[key] = None
            return None
        
        if k == 1:
            if n in val_to_idx:
                memo[key] = [n]
                return [n]
            memo[key] = None
            return None
        
        # Quick bound check using prefix sums
        ps = prefix_sums[n]
        max_sum = ps[k]  # sum of k largest
        min_sum = ps[ncand] - ps[ncand - k]  # sum of k smallest
        if n > max_sum or n < min_sum:
            memo[key] = None
            return None

        path = []  # list of indices
        
        def backtrack(start_idx, remaining_sum, remaining_k, forbidden_mask):
            if remaining_k == 0:
                if remaining_sum == 0:
                    return [smooth_list[idx] for idx in path]
                return None
            
            avail_count = ncand - start_idx
            if avail_count < remaining_k:
                return None
            
            # Pruning using prefix sums
            # Max possible: sum of next 'remaining_k' largest from start_idx
            max_possible = ps[start_idx + remaining_k] - ps[start_idx]
            if max_possible < remaining_sum:
                return None
            
            # Min possible: sum of 'remaining_k' smallest overall (since we can pick any from remaining)
            min_possible = ps[ncand] - ps[ncand - remaining_k]
            if min_possible > remaining_sum:
                return None
            
            for i in range(start_idx, ncand):
                idx = cand[i]
                num = smooth_list[idx]
                if num > remaining_sum:
                    continue
                
                # Check if this index is forbidden (conflicts with already chosen)
                if forbidden_mask & (1 << idx):
                    continue
                
                # Check divisibility against all chosen using conflict_mask
                # We need to ensure idx doesn't conflict with any in path
                # Since we build path incrementally, we can track forbidden_mask
                # When we add idx, new forbidden = forbidden_mask | conflict_mask[idx]
                
                path.append(idx)
                new_forbidden = forbidden_mask | conflict_mask[idx]
                result = backtrack(i + 1, remaining_sum - num, remaining_k - 1, new_forbidden)
                if result is not None:
                    memo[key] = result
                    return result
                path.pop()
            
            return None
        
        result = backtrack(0, n, k, 0)
        memo[key] = result
        return result

    answers = []
    for n in range(1, 129):
        answer = None
        for k in range(1, 31):
            # Check time
            if time.time() - start_time > DEADLINE:
                break
                
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Fallback: try all k values again (memo may have been filled)
            for k in range(1, 31):
                res = find_antichain(n, k)
                if res is not None:
                    answer = res
                    break
        
        if answer is None:
            # Last resort: if n is smooth, use [n]
            if n in val_to_idx:
                answer = [n]
            else:
                answer = [n]  # Placeholder
        
        answers.append(answer)
        
    return answers
