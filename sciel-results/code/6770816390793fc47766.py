import time

def construct():
    n = 256
    # Weights are 1/x
    # We want to maximize sum(1/x)
    
    # Precompute weights
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    # Sort indices by weight descending (which is just 1, 2, ..., n)
    # But we can just iterate 1 to n.
    
    best_set = []
    best_sum = 0.0
    
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
    deadline = start_time + 50.0 # Safety margin
    
    # We will use a recursive search.
    # State: current_set (list), current_sum (float), next_index (int)
    # To optimize, we can pass a set for O(1) lookup?
    # But checking 3-AP involves pairs.
    
    # Optimization: 
    # When adding x, we only need to check pairs (a,b) in S such that 2b-a=x.
    # This is equivalent to: for each b in S, check if 2b-x is in S.
    # This reduces O(k^2) to O(k) if S is a set.
    
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
            
        # Upper bound: current_sum + sum(weights[next_idx..n])
        # Precompute suffix sums of weights
        # suffix_sum[i] = sum(weights[i..n])
        
        # Pruning
        if current_sum + suffix_sum[next_idx] <= best_sum:
            return
            
        x = next_idx
        
        # Option 1: Include x
        if check_compatible(x, current_set):
            current_set.add(x)
            search(current_set, current_sum + weights[x], next_idx + 1)
            current_set.remove(x)
            
        # Option 2: Exclude x
        search(current_set, current_sum, next_idx + 1)
        
    # Precompute suffix sums
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    # Start search from empty set? Or from greedy?
    # Starting from empty set is safer for global optimum.
    # But greedy gives a good lower bound.
    # We can start search from empty set, but use best_sum from greedy for pruning.
    
    search(set(), 0.0, 1)
    
    return best_set
