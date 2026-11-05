def construct():
    from math import gcd
    import time
    import random

    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute conflicts
    # conflicts[x] is a set of y such that lcm(x,y) <= n
    conflicts = {x: set() for x in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            g = gcd(a, b)
            if (a // g) * b <= n:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Split into Core and Periphery
    # Core: small numbers, high connectivity
    # Periphery: large numbers, low connectivity
    core_limit = 20
    core = [x for x in candidates if x <= core_limit]
    periphery = [x for x in candidates if x > core_limit]
    
    # 1. Generate all independent sets in the Core
    # We use a recursive backtracking to enumerate independent sets.
    # To speed up, we can order core nodes by degree (descending) or just index.
    # Let's just use the list order.
    
    core_sets = []
    
    def gen_core_sets(idx, current_set, available):
        if idx == len(core):
            core_sets.append(frozenset(current_set))
            return
        
        x = core[idx]
        # Option 1: Exclude x
        gen_core_sets(idx + 1, current_set, available)
        
        # Option 2: Include x
        # Check if x conflicts with any in current_set
        if not (conflicts[x] & current_set):
            # Remove x and its neighbors from available for subsequent steps?
            # Actually, since we are just building a set, we just need to ensure
            # future additions don't conflict with x.
            # The recursion handles this by checking against current_set.
            # We don't strictly need 'available' if we check against current_set,
            # but passing a reduced list of candidates for future indices
            # that are not neighbors of x can prune the tree.
            
            # For simplicity and correctness first, we just recurse.
            # Optimization: only consider future nodes that are not neighbors of x.
            # But the index-based recursion doesn't easily allow skipping neighbors
            # unless we filter the core list dynamically.
            
            # Let's stick to the simple check.
            gen_core_sets(idx + 1, current_set | {x}, available)

    # This simple recursion might be slow if core is large.
    # Core size 19 (2..20). 2^19 = 524,288.
    # With pruning, it should be much faster.
    # Let's optimize by sorting core by degree descending?
    # If we pick high-degree nodes first, we prune more.
    
    core_sorted = sorted(core, key=lambda x: len(conflicts[x]), reverse=True)
    
    def gen_core_sets_opt(idx, current_set):
        if idx == len(core_sorted):
            core_sets.append(frozenset(current_set))
            return
        
        x = core_sorted[idx]
        # Exclude
        gen_core_sets_opt(idx + 1, current_set)
        
        # Include
        if not (conflicts[x] & current_set):
            gen_core_sets_opt(idx + 1, current_set | {x})

    gen_core_sets_opt(0, frozenset())
    
    # 2. For each core set, solve the periphery problem
    best_sum = 0.0
    best_set = set()
    
    # Pre-sort periphery by weight (1/x) descending for greedy
    periphery_sorted = sorted(periphery, key=lambda x: 1.0/x, reverse=True)
    
    def solve_periphery(available_periphery, core_set):
        # available_periphery is a list of nodes in periphery that do not conflict with core_set
        # We need to find MWIS in the subgraph induced by available_periphery.
        
        # Greedy initialization
        s = set()
        for x in available_periphery:
            if not (conflicts[x] & s):
                s.add(x)
        
        # Local search on periphery
        # Since periphery is sparse and large nodes, 1-swap is often sufficient.
        improved = True
        while improved:
            improved = False
            # Try adding
            for x in available_periphery:
                if x not in s and not (conflicts[x] & s):
                    s.add(x)
                    improved = True
                    break
            
            # Try 1-swap
            for x in list(s):
                rem = s - {x}
                for y in available_periphery:
                    if y in s: continue
                    if not (conflicts[y] & rem):
                        if 1.0/y > 1.0/x:
                            s = rem | {y}
                            improved = True
                            break
                if improved: break
            
            # Try 2-swap (optional, might be slow if done for every core set)
            # Given the number of core sets, let's stick to 1-swap for speed.
            # Or a very limited 2-swap.
            
        return s, sum(1.0/x for x in s)

    # Iterate through core sets
    # To speed up, we can bound the core sum.
    # Max possible periphery sum is sum(1/x for x in periphery).
    # If core_sum + max_periphery_sum <= best_sum, skip.
    max_per_sum = sum(1.0/x for x in periphery)
    
    for cs in core_sets:
        core_sum = sum(1.0/x for x in cs)
        if core_sum + max_per_sum <= best_sum:
            continue
            
        # Filter periphery
        # A node p is valid if it doesn't conflict with any node in cs
        valid_per = [p for p in periphery if not (conflicts[p] & cs)]
        
        if not valid_per:
            total_sum = core_sum
            total_set = set(cs)
        else:
            # Solve periphery
            p_set, p_sum = solve_periphery(valid_per, cs)
            total_sum = core_sum + p_sum
            total_set = cs | p_set
            
        if total_sum > best_sum:
            best_sum = total_sum
            best_set = total_set
            
    return list(best_set)
