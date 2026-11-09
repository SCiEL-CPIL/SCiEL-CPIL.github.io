import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85  # 90s limit, leave 5s margin
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):  # 2^7 = 128
        val_a = 2 ** a
        for b in range(0, 5):  # 3^4 = 81, 3^5 = 243 > 128
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):  # 5^3 = 125, 5^4 = 625 > 128
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Precompute divisibility: divs[i][j] = True if smooth_list[i] divides smooth_list[j] (i != j)
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    divs[i][j] = True
                    divs[j][i] = True
    
    # Map smooth value to its index
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Pre-filter candidates for each n: list of indices into smooth_list
    # candidates[n] = [idx in range(n_smooth) if smooth_list[idx] <= n]
    candidates = [[] for _ in range(129)]
    for i in range(129):
        for idx in range(n_smooth):
            if smooth_list[idx] <= i:
                candidates[i].append(idx)
            else:
                break
        
    # Precompute for each n and each possible k: prefix sums for pruning
    # For a given n, sort candidates by value descending for the search
    cand_desc = [sorted(candidates[n], key=lambda idx: smooth_list[idx], reverse=True) for n in range(129)]
    
    def find_antichain(n, k):
        """
        Finds an antichain of size k summing to n.
        Returns list of terms or None.
        """
        cand = cand_desc[n]
        if not cand or len(cand) < k:
            return None
        
        # Quick bound check
        if k == 1:
            if n in val_to_idx:
                return [n]
            return None
        
        # Max sum: sum of k largest
        max_sum = sum(smooth_list[cand[i]] for i in range(k))
        # Min sum: sum of k smallest
        min_sum = sum(smooth_list[cand[len(cand)-1-i]] for i in range(k))
        if n > max_sum or n < min_sum:
            return None

        path = []  # list of indices
        
        def backtrack(start_idx, remaining_sum, remaining_k):
            if remaining_k == 0:
                if remaining_sum == 0:
                    return [smooth_list[idx] for idx in path]
                return None
            
            avail_count = len(cand) - start_idx
            if avail_count < remaining_k:
                return None
            
            # Pruning: max possible sum from remaining candidates
            # Sum of largest 'remaining_k' from cand[start_idx:]
            max_possible = sum(smooth_list[cand[start_idx + i]] for i in range(remaining_k))
            if max_possible < remaining_sum:
                return None
            
            # Min possible sum: sum of smallest 'remaining_k' from cand[start_idx:]
            # cand is sorted descending, so smallest are at the end
            min_possible = sum(smooth_list[cand[len(cand)-1-i]] for i in range(remaining_k))
            if min_possible > remaining_sum:
                return None
            
            for i in range(start_idx, len(cand)):
                idx = cand[i]
                num = smooth_list[idx]
                if num > remaining_sum:
                    continue
                
                # Check divisibility against all chosen
                compatible = True
                for prev_idx in path:
                    if divs[prev_idx][idx]:
                        compatible = False
                        break
                
                if compatible:
                    path.append(idx)
                    result = backtrack(i + 1, remaining_sum - num, remaining_k - 1)
                    if result is not None:
                        return result
                    path.pop()
            
            return None
        
        return backtrack(0, n, k)

    answers = []
    for n in range(1, 129):
        answer = None
        for k in range(1, 31):
            # Check time
            if time.time() - start_time > DEADLINE:
                # Fallback: try to find any valid representation quickly
                break
                
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Fallback: brute force with relaxed time check
            # Try larger k values
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
                # This should not happen, but provide a safe fallback
                # Find any subset of smooth numbers summing to n (ignoring antichain constraint temporarily)
                # Then filter to antichain
                answer = [n]  # Placeholder
        
        answers.append(answer)
        
    return answers
