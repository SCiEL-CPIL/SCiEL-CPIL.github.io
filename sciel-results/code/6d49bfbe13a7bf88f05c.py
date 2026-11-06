def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    target = Fraction(5, 31)
    
    # Known solution with max denom 1736
    # 5/31 = 1/8 + 1/28 + 1/1736
    best_solution = [8, 28, 1736]
    best_max = 1736
    
    # Try to find a solution with max denom < 1736
    # We search for max_denom in range(100, 1736)
    # But doing a full search for each is too slow.
    # Instead, we do a single DFS search with a pruning bound on the max denominator.
    # We try to find ANY solution with max denom < 1736.
    
    def search(rem, min_d, max_d_allowed, used, path, depth):
        nonlocal best_solution, best_max
        if time.time() - start_time > 50:
            return
        
        if rem == 0:
            if len(path) > 0:
                curr_max = max(path)
                if curr_max < best_max:
                    best_max = curr_max
                    best_solution = path.copy()
            return
        
        if depth > 6: # Limit depth
            return
            
        # Pruning: if rem is small, we need large denominators
        # 1/min_d + 1/(min_d+1) + ... approx 1/min_d * (terms_left)
        # If 1/min_d * (6 - depth) < rem, we can't reach rem with remaining terms
        # Actually, max sum with k terms starting from min_d is roughly k/min_d
        # If rem > k/min_d, we need smaller denominators, but we are constrained by min_d.
        # So if rem > (6 - depth) / min_d, we fail.
        if rem > (6 - depth) / min_d:
            return
        
        # Find possible denominators d
        # d must be >= min_d
        # d must be <= max_d_allowed
        # 1/d < rem  => d > 1/rem
        # Also we need to leave a positive remainder for the rest
        # rem - 1/d > 0 => 1/d < rem => d > 1/rem
        # Also, the remaining terms must be able to sum to rem - 1/d
        # Max sum of remaining (k-1) terms starting from d+1 is (k-1)/(d+1)
        # So we need rem - 1/d <= (k-1)/(d+1)
        # This gives a lower bound for d.
        
        # Simple bound: d > 1/rem
        d_min = max(min_d, (rem.denominator // rem.numerator) + 1)
        
        # Upper bound: 1/d >= rem / (6 - depth) ? No.
        # We need to leave enough for the rest.
        # rem - 1/d <= (6 - depth - 1) / (d+1)
        # rem - 1/d <= (5 - depth) / (d+1)
        # rem(d+1) - (d+1) <= (5 - depth)d
        # rem*d + rem - d - 1 <= (5 - depth)d
        # rem - 1 <= d(5 - depth - rem)
        # This is getting messy. Let's just iterate d from d_min to max_d_allowed.
        
        for d in range(d_min, max_d_allowed + 1):
            if d in used:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                if len(path) + 1 > 0:
                    curr_max = max(path + [d])
                    if curr_max < best_max:
                        best_max = curr_max
                        best_solution = path + [d]
                continue
            
            used.add(d)
            search(new_rem, d + 1, max_d_allowed, used, path + [d], depth + 1)
            used.remove(d)
            
            if time.time() - start_time > 55:
                return

    # Try to find a solution with max denom < 1736
    # We can't just pass 1735 as max_d_allowed and expect it to find a small one quickly.
    # But if a solution exists, it will likely have small denominators.
    # Let's try max_d_allowed = 1735
    search(target, 1, 1735, set(), [], 0)
    
    return best_solution
