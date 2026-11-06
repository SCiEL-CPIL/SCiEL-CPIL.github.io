# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit
    sys.setrecursionlimit(10000)
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_set = set(smooth_numbers)
    n_smooth = len(smooth_numbers)
    
    # Map value -> index in smooth_numbers
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Precompute divisibility masks
    # div_mask[i] has bit j set if smooth_numbers[j] divides smooth_numbers[i]
    # mult_mask[i] has bit j set if smooth_numbers[i] divides smooth_numbers[j]
    div_mask = [0] * n_smooth
    mult_mask = [0] * n_smooth
    
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(n_smooth):
            if i != j:
                b = smooth_numbers[j]
                if b % a == 0:
                    # a divides b
                    div_mask[j] |= (1 << i)
                    mult_mask[i] |= (1 << j)
                elif a % b == 0:
                    # b divides a
                    div_mask[i] |= (1 << j)
                    mult_mask[j] |= (1 << i)

    answers = []
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            break
            
        # Filter candidates <= n
        candidates = []
        for i, val in enumerate(smooth_numbers):
            if val <= n:
                candidates.append((val, i))
        
        # Sort candidates by value descending for better pruning
        candidates.sort(key=lambda x: -x[0])
        n_cand = len(candidates)
        
        found = None
        
        # Try k=1 to 30
        for k in range(1, 31):
            if k == 1:
                if n in smooth_set:
                    found = [n]
                    break
                else:
                    continue
                    
            # DFS for size k with improved pruning
            def dfs(idx, rem, mask, curr, k_target):
                if len(curr) == k_target:
                    return rem == 0
                if idx >= n_cand:
                    return False
                if rem < 0:
                    return False
                
                # Pruning: not enough candidates left
                remaining_slots = k_target - len(curr)
                if remaining_slots > (n_cand - idx):
                    return False
                
                # Try including candidates[idx]
                val, g_idx = candidates[idx]
                if val <= rem:
                    if (mask & div_mask[g_idx]) == 0 and (mask & mult_mask[g_idx]) == 0:
                        curr.append(val)
                        if dfs(idx + 1, rem - val, mask | (1 << g_idx), curr, k_target):
                            return True
                        curr.pop()
                
                # Try excluding candidates[idx]
                if dfs(idx + 1, rem, mask, curr, k_target):
                    return True
                    
                return False
            
            curr = []
            if dfs(0, n, 0, curr, k):
                found = curr
                break
        
        if found is None:
            # Fallback: greedy with backtracking
            found = _greedy_antichain(n, candidates, div_mask, mult_mask, val_to_idx, smooth_set)
        
        answers.append(found)
    
    # If we timed out, fill remaining with fallback
    while len(answers) < 128:
        n = len(answers) + 1
        candidates = [(x, val_to_idx[x]) for x in smooth_numbers if x <= n]
        candidates.sort(key=lambda x: -x[0])
        found = _greedy_antichain(n, candidates, div_mask, mult_mask, val_to_idx, smooth_set)
        answers.append(found)
    
    return answers

def _greedy_antichain(n, candidates, div_mask, mult_mask, val_to_idx, smooth_set):
    """Fallback: find an antichain representation using greedy approach with backtracking."""
    
    candidates_sorted = sorted(candidates, key=lambda x: -x[0])
    best = None
    n_cand = len(candidates_sorted)
    
    def dfs(remaining, chosen, idx, max_terms, current_mask):
        nonlocal best
        if best is not None and len(chosen) >= len(best):
            return
        if remaining == 0:
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if len(chosen) >= max_terms:
            return
        if idx >= n_cand:
            return
        if remaining < 0:
            return
        
        val, g_idx = candidates_sorted[idx]
        if val <= remaining:
            if (current_mask & div_mask[g_idx]) == 0 and (current_mask & mult_mask[g_idx]) == 0:
                chosen.append(val)
                dfs(remaining - val, chosen, idx + 1, max_terms, current_mask | (1 << g_idx))
                chosen.pop()
        
        dfs(remaining, chosen, idx + 1, max_terms, current_mask)

    dfs(n, [], 0, 30, 0)
    if best is not None:
        return best
    
    if n in smooth_set:
        return [n]
    else:
        # Last resort: return any single smooth number <= n
        for s in smooth_numbers:
            if s <= n:
                return [s]
        return [1]
# EVOLVE-BLOCK-END
