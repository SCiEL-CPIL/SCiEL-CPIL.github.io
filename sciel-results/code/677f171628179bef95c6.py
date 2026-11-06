import time
from math import gcd

def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(2, N + 1):
        weights[i] = 1.0 / i
        
    # Precompute conflict masks using bitsets
    # Bit i corresponds to number i + 2
    masks = [0] * (N + 1)
    for v in range(2, N + 1):
        m = 0
        for u in range(2, N + 1):
            if u == v:
                continue
            g = gcd(v, u)
            if v * u <= 256 * g:
                m |= (1 << (u - 2))
        masks[v] = m
        
    # Weight array indexed by bit position (0..254)
    weights_bit = [0.0] * (N - 1)
    for p in range(N - 1):
        v = p + 2
        weights_bit[p] = 1.0 / v
        
    best_score = 0.0
    best_set = []
    
    # Initial candidates: all bits set for 2..256
    initial_mask = (1 << (N - 1)) - 1
    
    # Heuristic initialization to provide a strong lower bound
    # Candidate 1: {2} U {odd in [129, 256]}
    # Rationale: 2 conflicts with all evens. Odds in [129, 256] are large enough 
    # that their pairwise lcm is likely > 256.
    heuristic_set = [2] + [x for x in range(129, 257, 2)]
    
    # Verify validity of heuristic
    valid = True
    for i in range(len(heuristic_set)):
        for j in range(i+1, len(heuristic_set)):
            a, b = heuristic_set[i], heuristic_set[j]
            if a * b <= 256 * gcd(a, b):
                valid = False
                break
        if not valid:
            break
            
    if valid:
        best_score = sum(1.0/x for x in heuristic_set)
        best_set = list(heuristic_set)
    else:
        # Fallback: Greedy smallest first
        greedy_set = []
        for x in range(2, N + 1):
            # Check if x conflicts with any in greedy_set
            # We can use masks for speed, but greedy_set is small initially
            ok = True
            for y in greedy_set:
                if x * y <= 256 * gcd(x, y):
                    ok = False
                    break
            if ok:
                greedy_set.append(x)
        if greedy_set:
            best_score = sum(1.0/x for x in greedy_set)
            best_set = list(greedy_set)
            
    # Branch and Bound Solver
    # We use a recursive search with bitset masks.
    # To handle Python recursion limits and overhead, we implement a custom stack or 
    # rely on the fact that depth is bounded by N=255. 
    # However, standard recursion in Python can be slow. 
    # Let's try to optimize the bound calculation.
    
    # Precompute sum of weights for all possible masks? No, 2^255 is too big.
    # We compute bound(mask) on the fly.
    # Optimization: If mask has few bits, summing is fast.
    # We can also cache bound(mask) if we revisit states (unlikely in B&B tree).
    
    # A tighter bound: 
    # The simple sum of weights is valid.
    # Can we do better? 
    # For any independent set, no two nodes can be adjacent.
    # A common bound is the sum of weights of a maximal independent set found greedily? 
    # No, that's a lower bound.
    # Upper bound: sum of weights of all nodes is valid.
    # Another upper bound: For each connected component, the max weight independent set is <= sum of weights in that component.
    # Since the graph is sparse, components might be small?
    # Actually, the graph of conflicts for [2, 256] is quite connected near the small numbers.
    
    # Let's stick with the sum bound but optimize the function.
    
    def bound(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights_bit[idx]
            m ^= lsb
        return s
        
    # We will use a recursive function.
    # To avoid deep recursion issues, we might need to increase limit, 
    # but depth is max 255, so default 1000 is fine.
    
    # To speed up, we can order the branching.
    # Branching on the node with the highest weight (smallest number) usually prunes faster
    # because it resolves the "expensive" decisions early.
    
    # We also add a "dominance" check:
    # If node u is a neighbor of node v, and weight[u] <= weight[v] (which is true if u > v),
    # and N(v) subset N(u) U {v}? 
    # If v dominates u (v is smaller, weight higher, and conflicts with everything u conflicts with plus u itself),
    # then in any optimal solution, if u is picked, v can replace it?
    # Actually, if N(u) \ {v} subset N(v) \ {u}, then v is "better" than u.
    # If we pick u, we exclude N(u). If we pick v, we exclude N(v).
    # If N(u) subset N(v) U {v}, then picking v excludes a superset of what picking u excludes (mostly).
    # Wait, if N(u) subset N(v), then picking v is more restrictive. 
    # But v has higher weight.
    # This is complex. Let's stick to simple B&B.
    
    # One optimization: 
    # If the current best_score + bound(mask) <= best_score, prune.
    
    # Let's use a global counter to check time.
    
    import sys
    sys.setrecursionlimit(5000)
    
    def solve(mask, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
            
        # Upper bound
        ub = current_score + bound(mask)
        if ub <= best_score:
            return
            
        # Find the node with the smallest index (highest weight) in the mask
        # This is the lowest set bit.
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        v = idx + 2
        
        # Option 1: Include v
        # Remove v and its neighbors from the mask
        # neighbors_mask = masks[v]
        # new_mask = mask & ~(neighbors_mask | (1 << idx))
        new_mask_include = mask & ~(masks[v] | (1 << idx))
        
        # Heuristic: Include branch first? 
        # Since v has the highest weight among available, including it is likely good.
        # This might find a good solution faster, improving the bound for other branches.
        solve(new_mask_include, current_score + weights_bit[idx], current_set + [v])
        
        # Option 2: Exclude v
        new_mask_exclude = mask & ~(1 << idx)
        solve(new_mask_exclude, current_score, current_set)
        
    # Run the solver
    solve(initial_mask, 0.0, [])
    
    return best_set
