import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Try multiple strategies and return the best
    best_result = []
    
    def is_valid_set(chosen):
        """Check if the set satisfies the cardinality-separated subset sum property."""
        if not chosen:
            return True
        n = len(chosen)
        # S[k] = set of all subset sums of size k
        S = [set() for _ in range(n + 1)]
        S[0].add(0)
        for x in chosen:
            for k in range(n, 0, -1):
                for s in S[k-1]:
                    S[k].add(s + x)
        # Check that sums of different cardinalities are disjoint
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True
    
    def greedy_add(order, chosen, S, current_max_k):
        """Try to add elements from order to chosen, maintaining validity."""
        nonlocal best_result
        chosen = list(chosen)
        S = [set(s) for s in S]
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try adding x
            new_chosen = chosen + [x]
            new_max_k = current_max_k + 1
            new_S = [set(s) for s in S]
            
            # Compute new subset sums
            for k in range(new_max_k, 0, -1):
                if k - 1 >= 0 and k - 1 < len(S):
                    new_sums = set(s + x for s in S[k-1])
                    new_S[k] = new_S[k] | new_sums
            
            # Check disjointness: for all pairs of different cardinalities, sums must be disjoint
            valid = True
            for i in range(new_max_k + 1):
                for j in range(i + 1, new_max_k + 1):
                    if new_S[i] & new_S[j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                chosen = new_chosen
                S = new_S
                current_max_k = new_max_k
                if len(chosen) > len(best_result):
                    best_result = list(chosen)
        
        return chosen, S, current_max_k
    
    # Strategy 1: Try ascending order (smallest first)
    order_asc = list(range(1, 257))
    S_init = [set() for _ in range(61)]
    S_init[0].add(0)
    chosen, S, max_k = greedy_add(order_asc, [], S_init, 0)
    
    # Strategy 2: Try descending order (largest first)
    order_desc = list(range(256, 0, -1))
    S_init2 = [set() for _ in range(61)]
    S_init2[0].add(0)
    chosen2, S2, max_k2 = greedy_add(order_desc, [], S_init2, 0)
    
    # Strategy 3: Try a geometric-like sequence
    # Start with small numbers, then jump
    order_geo = []
    val = 1
    while val <= 256:
        order_geo.append(val)
        val *= 2
    # Fill in remaining
    for v in range(1, 257):
        if v not in order_geo:
            order_geo.append(v)
    S_init3 = [set() for _ in range(61)]
    S_init3[0].add(0)
    chosen3, S3, max_k3 = greedy_add(order_geo, [], S_init3, 0)
    
    # Strategy 4: Try powers of 2 first, then fill
    order_p2 = [2**i for i in range(1, 9) if 2**i <= 256]
    for v in range(1, 257):
        if v not in order_p2:
            order_p2.append(v)
    S_init4 = [set() for _ in range(61)]
    S_init4[0].add(0)
    chosen4, S4, max_k4 = greedy_add(order_p2, [], S_init4, 0)
    
    # Return the best result found
    if not best_result:
        # Fallback: return a small valid set
        test_set = [1, 2, 4, 8, 16, 32, 64, 128]
        if is_valid_set(test_set):
            return test_set
        return [1, 2, 4, 8, 16, 32]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_result]
