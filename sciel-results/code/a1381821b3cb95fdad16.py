import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    best_score = 0.0
    best_set = []
    
    in_S = [False] * (N + 1)
    S_list = []
    
    # Greedy initialization to get a decent starting bound
    # This helps pruning significantly
    for i in range(1, N + 1):
        valid = True
        for a in S_list:
            # Check if a, b, i form AP with i largest
            # 2b = a + i => b = (a+i)/2
            s = a + i
            if s % 2 == 0:
                b = s // 2
                if in_S[b]:
                    valid = False
                    break
        if valid:
            in_S[i] = True
            S_list.append(i)
            
    if S_list:
        best_score = sum(weights[x] for x in S_list)
        best_set = list(S_list)
        
    # Reset for B&B
    in_S = [False] * (N + 1)
    S_list = []
    
    def is_valid_add(x):
        # Check if adding x creates a 3-AP with existing elements
        # Since x is the largest in the current prefix, we only check patterns (a, b, x)
        # where a, b are in S and a < b < x.
        # This is equivalent to checking if (a+x)/2 is in S for any a in S.
        for a in S_list:
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if in_S[b]:
                    return False
        return True

    def get_tight_upper_bound(idx, current_sum):
        # Calculate an upper bound by summing weights of all remaining elements
        # that are individually compatible with the current set S.
        # This ignores 3-APs formed by pairs of remaining elements,
        # but provides a much tighter bound than the simple suffix sum.
        ub = current_sum
        for j in range(idx, N + 1):
            if is_valid_add(j):
                ub += weights[j]
        return ub

    def dfs(idx, current_sum):
        nonlocal best_score, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning with tight upper bound
        ub = get_tight_upper_bound(idx, current_sum)
        if ub <= best_score:
            return
            
        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_set = list(S_list)
            return
            
        x = idx
        
        # Option 1: Include x
        if is_valid_add(x):
            in_S[x] = True
            S_list.append(x)
            
            dfs(idx + 1, current_sum + weights[x])
            
            S_list.pop()
            in_S[x] = False
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum)

    dfs(1, 0.0)
    
    return list(best_set)
