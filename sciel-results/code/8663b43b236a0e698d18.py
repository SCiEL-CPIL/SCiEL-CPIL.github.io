import time
from math import gcd

def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    
    # 1. Precompute conflicts
    # Conflict if lcm(a, b) <= 256
    # lcm(a, b) = a * b // gcd(a, b)
    # So a * b <= 256 * gcd(a, b)
    
    N = 256
    weights = [0.0] * (N + 1)
    for i in range(2, N + 1):
        weights[i] = 1.0 / i
        
    # Adjacency list for conflicts
    # We only care about nodes 2..256
    # Let's map them to indices 0..254 for easier array handling? 
    # Or just keep 1..256 and ignore 0, 1.
    
    adj = [[] for _ in range(N + 1)]
    
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            # Check lcm condition
            # a * b / g <= 256  <=> a * b <= 256 * g
            if a * b <= 256 * g:
                adj[a].append(b)
                adj[b].append(a)
    
    # 2. Branch and Bound for Maximum Weight Independent Set
    # We want to select a subset S such that no two elements in S are connected by an edge in adj.
    # Maximize sum(weights[x] for x in S).
    
    # To make Branch and Bound efficient, we should order nodes.
    # A common heuristic for MWIS is to process nodes in an order that allows for good bounds.
    # However, since weights are 1/x, smaller x have higher weight.
    # But smaller x have MORE conflicts (higher degree).
    # Larger x have fewer conflicts and lower weight.
    
    # Let's use a recursive backtracking with bounds.
    # State: 
    #   - current set of selected nodes (or just the sum so far)
    #   - set of available nodes (candidates that don't conflict with selected)
    #   - upper bound on the sum of weights of any independent set from available nodes
    
    # A simple upper bound: sum of weights of all available nodes.
    # This is valid because an independent set is a subset of available nodes.
    # It might be loose, but let's see.
    # A tighter bound: Greedy coloring bound? Or just sum of top k weights?
    # Given N=255, sum of all weights is H_256 - 1 ~ 6.
    # The optimal value is around 1.06.
    # The bound "sum of all available" will be very loose initially.
    
    # Better approach: 
    # Sort nodes. 
    # Let's try a different formulation.
    # Since the graph is not dense (high numbers are isolated from each other mostly),
    # we can use a simple recursive search that branches on inclusion/exclusion.
    
    # To ensure speed, we can use the fact that if a node has no neighbors in the remaining set,
    # we can include it immediately (if it improves score) or exclude it.
    # Actually, for MWIS, if a node is isolated in the subgraph, we should definitely include it (since weight > 0).
    
    # Let's implement a standard Branch and Bound.
    
    # Precompute a static ordering? 
    # Or dynamic?
    
    # Let's try a simpler heuristic first:
    # The set {2} U {odd in [129, 256]} is a strong candidate.
    # Score ~ 0.5 + sum(1/x for x in 129..256 step 2)
    
    # Let's verify the structural insight:
    # If we pick any even number > 2, say 2k, it conflicts with 2 (lcm(2, 2k)=2k <= 256 if k<=128).
    # Wait, lcm(2, 2k) = 2k. If 2k <= 256, they conflict.
    # So if we pick 2, we cannot pick any other even number <= 256.
    # Thus, if 2 is in the set, all other elements must be odd.
    
    # What if we don't pick 2?
    # Then we can pick other evens.
    # But evens generally have more conflicts than odds?
    # lcm(a, b) <= 256.
    # If a, b are both large (e.g., > 128), lcm(a, b) is likely > 256 unless they share factors.
    # Specifically, if a, b > 128, and gcd(a, b) = 1, lcm = ab > 128*128 = 16384 > 256.
    # If gcd(a, b) = g, lcm = ab/g.
    # For lcm <= 256, we need ab <= 256g.
    # Since a, b > 128, ab > 16384.
    # So we need 16384 <= 256g => g >= 64.
    # So two numbers > 128 conflict only if their gcd is at least 64.
    
    # This suggests that the "large" part of the set can be dense if we pick numbers with large common factors.
    # But weights 1/x are small for large x.
    
    # The trade-off is between picking small numbers (high weight, many conflicts) and large numbers (low weight, few conflicts).
    
    # Let's run a specialized search.
    # We will branch on the smallest available number.
    
    # Optimization: 
    # If we decide to include a number x, we remove x and all its neighbors from the candidate set.
    # If we exclude x, we just remove x.
    
    # To make this fast, we need a good bound.
    # Bound: Sum of weights of all remaining candidates.
    # This is O(N) to compute if we maintain a list.
    
    # Let's use a bitset for the candidate set? 
    # N=256 fits in 4 uint64s or a Python int.
    
    # Let's use Python integers as bitsets.
    # Bit i represents number i+2? Or just index 0..254 for numbers 2..256?
    # Let's map number v (2..256) to bit index v-2.
    
    # Precompute conflict masks
    # mask[v] is an integer with bits set for all u such that lcm(v, u) <= 256.
    # Note: v does not conflict with itself, but we handle that by removing v from candidates.
    
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
        
    # Weight array indexed by bit position
    # Bit pos p corresponds to number p + 2
    weights_bit = [0.0] * (N - 1)
    for p in range(N - 1):
        v = p + 2
        weights_bit[p] = 1.0 / v
        
    best_score = 0.0
    best_set = []
    
    # Initial candidates: all bits set for 2..256
    # Bits 0 to 254
    initial_mask = (1 << (N - 1)) - 1
    
    def bound(mask):
        # Sum of weights of set bits in mask
        # This is expensive if done naively.
        # We can precompute sum for chunks?
        # Or just iterate. 255 iterations max.
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights_bit[idx]
            m ^= lsb
        return s
        
    # Recursive search
    # To speed up, we can use a heuristic: 
    # Always branch on the node with the highest weight? 
    # Or the one with the highest degree?
    # Let's try branching on the smallest number available (highest weight).
    
    # However, standard MWIS B&B often uses coloring bounds.
    # Given the small N, a simple sum bound might work if the search tree is pruned well.
    
    # Let's add a time check inside the recursion.
    # Python recursion limit might be an issue if depth is 255.
    # sys.setrecursionlimit(300)
    
    import sys
    sys.setrecursionlimit(10000)
    
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
            
        # Heuristic: Find the "best" node to branch on.
        # Let's pick the node with the smallest index (highest weight) that is in the mask.
        # This tends to prune earlier because high-weight nodes are resolved first.
        
        # Find lowest set bit
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        v = idx + 2
        
        # Option 1: Include v
        # New mask: remove v and all neighbors of v
        # Neighbors of v are in masks[v].
        # We need to intersect with current mask.
        # new_mask = mask & ~(masks[v] | (1 << idx))
        # Note: masks[v] might contain bits outside current mask, so & mask is good.
        # Actually, masks[v] contains neighbors. We want to remove them AND v.
        # So remove_bits = masks[v] | (1 << idx)
        # new_mask = mask & ~remove_bits
        
        new_mask_include = mask & ~(masks[v] | (1 << idx))
        
        # We can also improve the bound for the "include" branch?
        # The bound function will recompute.
        
        solve(new_mask_include, current_score + weights_bit[idx], current_set + [v])
        
        # Option 2: Exclude v
        # New mask: remove v
        new_mask_exclude = mask & ~(1 << idx)
        
        solve(new_mask_exclude, current_score, current_set)
        
    # Start search
    # To help the solver, we can provide a good initial lower bound (best_score).
    # Let's use the heuristic set: {2} U {odd in [129, 256]}
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
        # Fallback to greedy if heuristic fails (shouldn't happen)
        best_score = 0.0
        best_set = []
        
    # Run the exact solver
    # The solver might be slow if the bound is weak.
    # Let's monitor time.
    
    # Optimization: If the bound is tight enough, it prunes.
    # Let's run it.
    
    # To avoid deep recursion overhead in Python, we could use an iterative approach or C extension, 
    # but we are limited to Python.
    # With N=255, the depth is at most 255.
    # The branching factor is 2.
    # Without good pruning, 2^255 is impossible.
    # The bound `sum of weights` is key.
    
    # Let's add a check: if the time is running out, return current best.
    
    solve(initial_mask, 0.0, [])
    
    return best_set
