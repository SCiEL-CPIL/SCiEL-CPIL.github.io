"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _build_conflict_masks(smooth):
    """Build conflict masks where mask[i] is a bitmask of indices that conflict with smooth[i]."""
    n = len(smooth)
    masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = smooth[i], smooth[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
    return masks

def _solve_target(n, smooth, conflict_masks, deadline, min_count=1, max_count=30):
    """Find minimum count antichain of smooth numbers summing to n."""
    candidates = [x for x in smooth if x <= n]
    if not candidates:
        return None
    
    # Build index mapping
    cand_to_idx = {val: i for i, val in enumerate(smooth)}
    cand_indices = [cand_to_idx[val] for val in candidates]
    
    # Sort by value descending for better pruning
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i])
    
    for count in range(min_count, max_count + 1):
        if time.time() > deadline:
            break
        if len(candidates) < count:
            break
        
        solution = None
        
        def dfs(start_idx, remaining, current_vals, current_idx_mask):
            nonlocal solution
            if solution is not None:
                return
            if time.time() > deadline:
                return
            
            if len(current_vals) == count:
                if remaining == 0:
                    solution = list(current_vals)
                return
            
            if remaining < 0:
                return
            
            needed = count - len(current_vals)
            if remaining < needed:
                return
            
            for i in range(start_idx, len(order)):
                if time.time() > deadline:
                    return
                
                ci = order[i]
                val = candidates[ci]
                if val > remaining:
                    continue
                
                idx = cand_indices[ci]
                
                # Check conflict with already chosen terms
                if current_idx_mask & conflict_masks[idx]:
                    continue
                
                current_vals.append(val)
                dfs(i + 1, remaining - val, current_vals, current_idx_mask | (1 << idx))
                current_vals.pop()
                
                if solution is not None:
                    return
        
        dfs(0, n, [], 0)
        
        if solution is not None:
            return solution
    
    return None

def construct():
    """
    Construct 128 rows corresponding to targets 1..128.
    Every row must contain 1..30 distinct positive 2^a*3^b*5^c terms summing to its target;
    no chosen term may divide another. Minimize the total number of summands.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s limit
    
    max_val = 128
    smooth = _generate_smooth(max_val)
    conflict_masks = _build_conflict_masks(smooth)
    
    # For each target, find the minimum number of terms
    answers = [None] * 129
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        solution = _solve_target(n, smooth, conflict_masks, deadline)
        answers[n] = solution
    
    # Return 128 rows for targets 1..128
    result = []
    for n in range(1, 129):
        if answers[n] is None:
            # Fallback: this should not happen
            if n in smooth:
                result.append([n])
            else:
                result.append([n])
        else:
            result.append(answers[n])
    
    return result
# EVOLVE-BLOCK-END
