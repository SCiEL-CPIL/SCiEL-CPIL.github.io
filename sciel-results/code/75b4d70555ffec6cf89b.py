import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    universe = set(range(1, 101))
    
    def find_violating_subset(candidates):
        cands = sorted(candidates)
        n = len(cands)
        if sum(cands) < 200:
            return None
        
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + cands[i]
            
        def dfs_pruned(idx, remaining, current_subset):
            if time.time() > deadline - 5:
                return None
            if remaining == 0:
                return current_subset
            if idx >= n:
                return None
            if remaining < 0:
                return None
            if suffix_sums[idx] < remaining:
                return None
            
            # Include
            if cands[idx] <= remaining:
                res = dfs_pruned(idx + 1, remaining - cands[idx], current_subset + [cands[idx]])
                if res:
                    return res
            
            # Exclude
            return dfs_pruned(idx + 1, remaining, current_subset)
            
        return dfs_pruned(0, 200, [])

    def find_disjoint_count(S):
        count = 0
        remaining = set(S)
        while True:
            if time.time() > deadline - 5:
                return count
            C = find_violating_subset(remaining)
            if C is None:
                break
            count += 1
            remaining -= set(C)
        return count

    # Greedy Initialization
    R_greedy = set()
    S_greedy = set(universe)
    while True:
        if time.time() > deadline - 20: break
        C = find_violating_subset(S_greedy)
        if C is None:
            break
        # Heuristic: remove element that appears in most disjoint sets?
        # Just remove max(C) for simplicity
        x_to_remove = max(C)
        R_greedy.add(x_to_remove)
        S_greedy.remove(x_to_remove)
        
    best_size = len(R_greedy)
    best_R = set(R_greedy)
    
    def solve(R, S):
        nonlocal best_R, best_size
        if time.time() > deadline - 5:
            return
        
        if len(R) >= best_size:
            return
            
        C = find_violating_subset(S)
        
        if C is None:
            if len(R) < best_size:
                best_size = len(R)
                best_R = set(R)
            return
        
        # Lower bound pruning
        lb = find_disjoint_count(S)
        if len(R) + lb >= best_size:
            return
        
        # Branch on elements in C
        # Sort C to try "better" removals first
        # Heuristic: try removing elements that are in many disjoint sets?
        # For now, just iterate
        for x in C:
            new_R = R | {x}
            new_S = S - {x}
            solve(new_R, new_S)
            
    solve(set(), set(universe))
    
    if best_R is None:
        best_R = R_greedy
        
    result_set = list(universe - best_R)
    return sorted(result_set)
