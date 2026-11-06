def construct():
    n = 256
    nums = list(range(n, 0, -1))
    
    def can_add(x, current_S):
        if not current_S:
            return True
        residues = {}
        for val in current_S:
            r = val % x
            residues[r] = residues.get(r, 0) + 1
        
        for b in current_S:
            rb = b % x
            target = (-rb) % x
            if target in residues:
                if target == rb:
                    if residues[target] >= 2:
                        return False
                else:
                    if residues[target] >= 1:
                        return False
        return True

    # Initial greedy
    S_greedy = []
    for x in nums:
        if can_add(x, S_greedy):
            S_greedy.append(x)
            
    best_size = len(S_greedy)
    best_set = list(S_greedy)
    
    S = []
    
    def dfs(idx, current_S):
        nonlocal best_size, best_set
        
        if len(current_S) + (n - idx) <= best_size:
            return
            
        if idx == n:
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_set = list(current_S)
            return

        x = nums[idx]
        
        # Option 1: Include x
        if can_add(x, current_S):
            current_S.append(x)
            dfs(idx + 1, current_S)
            current_S.pop()
            
        # Option 2: Exclude x
        dfs(idx + 1, current_S)

    dfs(0, S)
    
    return best_set

