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
    
    # Precompute partners for each x
    # partners[x] contains a < x such that (a+x) is even
    partners = [[] for _ in range(N + 1)]
    for x in range(2, N + 1):
        # a and x must have same parity
        start_a = 1 if x % 2 == 1 else 2
        for a in range(start_a, x, 2):
            partners[x].append(a)

    # Greedy initialization
    in_S_mask = 0
    S_list = []
    
    for i in range(1, N + 1):
        valid = True
        for a in partners[i]:
            b = (a + i) // 2
            if (in_S_mask >> b) & 1:
                valid = False
                break
        if valid:
            in_S_mask |= (1 << i)
            S_list.append(i)
            
    if S_list:
        best_score = sum(weights[x] for x in S_list)
        best_set = list(S_list)
        
    # Reset for B&B
    in_S_mask = 0
    S_list = []
    
    def is_valid_add(x):
        for a in partners[x]:
            b = (a + x) // 2
            if (in_S_mask >> b) & 1:
                return False
        return True

    def get_tight_upper_bound(idx, current_sum):
        ub = current_sum
        for j in range(idx, N + 1):
            # Check if j is valid with current S
            valid = True
            for a in partners[j]:
                b = (a + j) // 2
                if (in_S_mask >> b) & 1:
                    valid = False
                    break
            if valid:
                ub += weights[j]
        return ub

    def dfs(idx, current_sum):
        nonlocal best_score, best_set, in_S_mask, S_list
        
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
            in_S_mask |= (1 << x)
            S_list.append(x)
            
            dfs(idx + 1, current_sum + weights[x])
            
            S_list.pop()
            in_S_mask &= ~(1 << x)
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum)

    dfs(1, 0.0)
    
    return list(best_set)
