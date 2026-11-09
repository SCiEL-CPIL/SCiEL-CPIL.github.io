"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    import time
    
    n = 256
    # Split into Small and Large to exploit structure
    # Small: 2..12 (11 numbers)
    # Large: 13..256 (244 numbers)
    small_nums = list(range(2, 13))
    large_nums = list(range(13, 257))
    
    small_count = len(small_nums)
    large_count = len(large_nums)
    
    # Precompute weights
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    
    # Precompute conflicts
    # Small-Small
    small_conflict = [[False] * small_count for _ in range(small_count)]
    for i in range(small_count):
        for j in range(i + 1, small_count):
            if lcm(small_nums[i], small_nums[j]) <= n:
                small_conflict[i][j] = True
                small_conflict[j][i] = True
                
    # Large-Large
    large_conflict = [set() for _ in range(large_count)]
    for i in range(large_count):
        for j in range(i + 1, large_count):
            if lcm(large_nums[i], large_nums[j]) <= n:
                large_conflict[i].add(j)
                large_conflict[j].add(i)
                
    # Small-Large
    # small_large_block[i] is a set of large indices that conflict with small i
    small_large_block = [set() for _ in range(small_count)]
    for i in range(small_count):
        for j in range(large_count):
            if lcm(small_nums[i], large_nums[j]) <= n:
                small_large_block[i].add(j)
                
    # Pre-sort large indices by weight (descending)
    large_indices_sorted = sorted(range(large_count), key=lambda i: -large_weights[i])
    
    def solve_large(available_large_set):
        """
        Solve MWIS on the induced subgraph of large nodes in available_large_set.
        Returns (selected_set, weight).
        """
        if not available_large_set:
            return set(), 0.0
            
        selected = set()
        
        # Greedy initialization
        for idx in large_indices_sorted:
            if idx in available_large_set:
                if not (large_conflict[idx] & selected):
                    selected.add(idx)
                    
        current_weight = sum(large_weights[i] for i in selected)
        
        # Local Search: Add + Swap
        # Limit iterations to ensure speed
        max_iters = 5
        for _ in range(max_iters):
            improved = False
            
            # Try adding
            for idx in available_large_set:
                if idx not in selected:
                    if not (large_conflict[idx] & selected):
                        selected.add(idx)
                        current_weight += large_weights[idx]
                        improved = True
                        
            # Try swapping
            # Only consider removing nodes that allow a better addition
            # To speed up, we just do a simple pass
            removable_list = list(selected)
            for rem in removable_list:
                temp_selected = selected - {rem}
                best_gain = 0
                best_add = None
                for add in available_large_set:
                    if add not in temp_selected:
                        if not (large_conflict[add] & temp_selected):
                            gain = large_weights[add] - large_weights[rem]
                            if gain > best_gain:
                                best_gain = gain
                                best_add = add
                if best_add is not None and best_gain > 1e-15:
                    selected.discard(rem)
                    selected.add(best_add)
                    current_weight += best_gain
                    improved = True
                    
            if not improved:
                break
                
        return selected, current_weight

    best_score = 0.0
    best_result = []
    
    # Iterate over all subsets of Small numbers
    # There are 2^11 = 2048 subsets
    for mask in range(1 << small_count):
        # Check if this subset is a valid independent set in small graph
        valid = True
        selected_small = []
        for i in range(small_count):
            if mask & (1 << i):
                selected_small.append(i)
                for j in range(i + 1, small_count):
                    if mask & (1 << j):
                        if small_conflict[i][j]:
                            valid = False
                            break
                if not valid:
                    break
                    
        if not valid:
            continue
            
        small_weight = sum(small_weights[i] for i in selected_small)
        
        # Determine blocked large nodes
        blocked_large = set()
        for i in selected_small:
            blocked_large.update(small_large_block[i])
            
        available_large = set(range(large_count)) - blocked_large
        
        # Solve large part
        large_selected, large_weight = solve_large(available_large)
        
        total_weight = small_weight + large_weight
        
        if total_weight > best_score:
            best_score = total_weight
            # Construct result
            res = [small_nums[i] for i in selected_small]
            res.extend([large_nums[i] for i in large_selected])
            best_result = res

    best_result.sort()
    return best_result
# EVOLVE-BLOCK-END
