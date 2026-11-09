def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85
    
    candidates = list(range(2, 257))
    
    # Precompute conflict sets
    conflict = {}
    for x in range(2, 257):
        conflict[x] = set()
        for y in range(2, 257):
            if y == x:
                continue
            g = gcd(x, y)
            l = x * y // g
            if l <= 256:
                conflict[x].add(y)
    
    # Sort by conflict degree ascending, then by value ascending
    conflict_degree = {x: len(conflict[x]) for x in candidates}
    candidates.sort(key=lambda x: (conflict_degree[x], x))
    
    n = len(candidates)
    
    # Greedy initialization
    greedy_set = []
    greedy_sum = 0.0
    for x in candidates:
        compatible = True
        for y in greedy_set:
            if x in conflict[y]:
                compatible = False
                break
        if compatible:
            greedy_set.append(x)
            greedy_sum += 1.0 / x
    
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    # Suffix sums
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / candidates[i]
    
    call_count = [0]
    
    def dfs(idx, selected, current_sum):
        nonlocal best_sum, best_set
        
        call_count[0] += 1
        if time.time() > deadline:
            return
        
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected)
            return
        
        # Tight upper bound
        upper = current_sum
        for j in range(idx, n):
            c = candidates[j]
            compatible = True
            for y in selected:
                if c in conflict[y]:
                    compatible = False
                    break
            if compatible:
                upper += 1.0 / c
        
        if upper <= best_sum:
            return
        
        x = candidates[idx]
        
        # Try including x
        can_include = True
        for y in selected:
            if x in conflict[y]:
                can_include = False
                break
        
        if can_include:
            selected.append(x)
            dfs(idx + 1, selected, current_sum + 1.0 / x)
            selected.pop()
        
        # Try excluding x
        dfs(idx + 1, selected, current_sum)
    
    dfs(0, [], 0.0)
    
    if not best_set:
        return [2]
    return sorted(best_set)
