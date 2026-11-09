def _is_reciprocal_sum_free_fast(s):
    s_list = list(s)
    if not s_list:
        return True
    LCM = 5354228880
    for x in s_list:
        target = LCM // x
        others = [y for y in s_list if y != x]
        if not others:
            continue
        weights = [LCM // y for y in others]
        # Sort weights descending to find large sums quickly?
        # Actually, for subset sum, order doesn't matter for correctness, but for pruning in DFS it does.
        # For set of sums, we can just iterate.
        
        # Optimization: if sum(weights) < target, return False early for this x? No, continue to next x.
        # If sum(weights) < target, then x is safe.
        if sum(weights) < target:
            continue
            
        # Use a set of reachable sums
        reachable = {0}
        for w in weights:
            new_reachable = set()
            for r in reachable:
                nr = r + w
                if nr <= target:
                    new_reachable.add(nr)
            reachable.update(new_reachable)
            
        if target in reachable:
            return False
            
    return True
