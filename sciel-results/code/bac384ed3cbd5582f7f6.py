def construct():
    import time
    from fractions import Fraction
    
    target = Fraction(5, 31)
    deadline = time.time() + 50  # Safety margin
    
    # Known fallback
    best_solution = [8, 31, 248]
    
    # Search for a solution with max denominator < 248
    # We try to find a set of distinct denominators d_i <= C such that sum(1/d_i) = target
    
    def dfs(remaining, min_denom, current_set, C):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current_set
        if len(current_set) > 10: # Heuristic bound on terms
            return None
            
        # Lower bound for next denominator k:
        # 1/k <= remaining => k >= 1/remaining
        # k >= min_denom
        k_min = max(min_denom, int(1 / remaining) + 1)
        
        # Upper bound for next denominator k:
        # k <= C
        # Also, we need enough "room" for remaining terms.
        # If we pick k, the max sum we can get from k..C is sum(1/i for i in k..C).
        # If sum(1/i for i in k..C) < remaining, then no solution.
        
        for k in range(k_min, C + 1):
            new_rem = remaining - Fraction(1, k)
            if new_rem < 0:
                break
            if new_rem == 0:
                return current_set + [k]
            
            # Pruning: check if max possible sum from k+1..C is >= new_rem
            # Approximate sum 1/(k+1) + ... + 1/C ~ ln(C) - ln(k)
            # For small C, we can just check if 1/(k+1) + ... is possible
            # Simple prune: if new_rem > 1/(k+1) + 1/(k+2) + ... + 1/C
            # We can estimate this. If C is small, we can compute it or use a loose bound.
            # A loose bound: if new_rem > 1/(k+1) * (C-k), it might still be possible, but if new_rem > 1/(k+1) + 1/(k+2) ... 
            # Actually, just checking if 1/(k+1) < new_rem is not enough.
            # If new_rem > 0, we must pick at least one more.
            # If 1/(k+1) + ... + 1/C < new_rem, return None.
            
            # Efficient check:
            # If C - k is small, compute sum.
            # If C is large, use integral approximation.
            # Given C <= 248, we can just check if the sum of reciprocals from k+1 to C is >= new_rem.
            # But doing this in every step is slow.
            # Alternative: just recurse. The depth is small.
            
            res = dfs(new_rem, k + 1, current_set + [k], C)
            if res:
                return res
        return None

    # Try C from 10 to 247
    for C in range(10, 248):
        if time.time() > deadline:
            break
        res = dfs(target, 1, [], C)
        if res:
            return res
            
    return best_solution
