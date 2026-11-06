import time

def construct():
    n = 256
    # Weights are 1/x
    # We want to maximize sum(1/x)
    
    # Precompute weights
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    # Greedy initial solution
    curr_set = []
    curr_sum = 0.0
    for x in range(1, n + 1):
        valid = True
        # Check if x forms a 3-AP with any pair in curr_set
        # Since we add in increasing order, x is the largest.
        # AP is a, b, x with 2b = a + x => x = 2b - a
        for i in range(len(curr_set)):
            for j in range(i + 1, len(curr_set)):
                a = curr_set[i]
                b = curr_set[j]
                if 2 * b - a == x:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            curr_set.append(x)
            curr_sum += weights[x]
            
    best_set = list(curr_set)
    best_sum = curr_sum
    
    # Branch and Bound
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # Precompute suffix sums
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Better upper bound: greedy completion from next_idx
    # Given a set S and next index, greedily add elements to get an upper bound
    def greedy_upper_bound(S, next_idx):
        """Upper bound: greedily add remaining elements that are compatible."""
        S_set = set(S)
        bound = 0.0
        for x in range(next_idx, n + 1):
            # Check if x is compatible with S_set
            compatible = True
            for b in S_set:
                a = 2 * b - x
                if a in S_set:
                    compatible = False
                    break
            if compatible:
                S_set.add(x)
                bound += weights[x]
        return bound
    
    S = set(curr_set)
    
    def check_compatible(x, S):
        # x is larger than all in S? No, in B&B we might skip some.
        # But we iterate candidates in order.
        # Actually, if we skip some, the order is still increasing.
        # So x > all y in S.
        # So x can only be the 3rd term.
        # AP: a, b, x. 2b = a + x => a = 2b - x.
        # Check if a is in S.
        for b in S:
            a = 2 * b - x
            if a in S:
                return False
        return True
    
    def search(current_set, current_sum, next_idx):
        nonlocal best_set, best_sum
        
        if time.time() > deadline:
            return
            
        # Update best
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
            
        if next_idx > n:
            return
            
        # Upper bound: current_sum + greedy completion
        # Use greedy upper bound for tighter pruning
        upper = current_sum + greedy_upper_bound(current_set, next_idx)
        
        # Pruning
        if upper <= best_sum:
            return
            
        x = next_idx
        
        # Option 1: Include x
        if check_compatible(x, current_set):
            current_set.add(x)
            search(current_set, current_sum + weights[x], next_idx + 1)
            current_set.remove(x)
            
        # Option 2: Exclude x
        search(current_set, current_sum, next_idx + 1)
    
    # Start search from empty set, but use best_sum from greedy for pruning
    search(set(), 0.0, 1)
    
    # Local search: try to swap elements to improve
    if time.time() < deadline - 5:
        S = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each element not in S
            for x in range(1, n + 1):
                if x in S:
                    continue
                if check_compatible(x, S):
                    S.add(x)
                    best_set = list(S)
                    best_sum += weights[x]
                    improved = True
                    break
            # Try swapping: remove y, add x where x > y
            for y in list(S):
                S.remove(y)
                best_sum -= weights[y]
                for x in range(y + 1, n + 1):
                    if x in S:
                        continue
                    if check_compatible(x, S):
                        S.add(x)
                        best_sum += weights[x]
                        if best_sum > 0:  # valid
                            best_set = list(S)
                            improved = True
                            break
                if not improved:
                    S.add(y)
                    best_sum += weights[y]
    
    return best_set
