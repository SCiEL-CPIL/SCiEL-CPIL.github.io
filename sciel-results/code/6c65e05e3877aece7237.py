import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precompute t-subsets and map to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_count = len(all_t_subsets)
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute t_sub_to_blocks
    # For each t-subset, find all k-blocks containing it
    t_sub_to_blocks = [[] for _ in range(t_sub_count)]
    
    # To do this efficiently, we iterate over t-subsets and generate extensions
    for i, t_sub in enumerate(all_t_subsets):
        t_sub_set = set(t_sub)
        remaining_points = [p for p in range(v) if p not in t_sub_set]
        # Generate (k-t) subsets from remaining_points
        for ext in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(t_sub + ext))
            # We don't have a global block index yet, but we can store the block tuple
            # Or we can map block tuple to index later.
            # Storing tuples in a list of lists is fine.
            t_sub_to_blocks[i].append(block)

    # 3. Greedy Covering
    cover_count = [0] * t_sub_count
    solution_blocks = []
    uncovered_count = t_sub_count
    
    # Helper to get t-subsets of a block
    def get_block_t_subs(block):
        return list(itertools.combinations(block, t))

    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Find first uncovered t-subset
        target_t_idx = -1
        for i in range(t_sub_count):
            if cover_count[i] == 0:
                target_t_idx = i
                break
        if target_t_idx == -1:
            break
            
        candidates = t_sub_to_blocks[target_t_idx]
        
        best_block = None
        best_gain = -1
        
        for block in candidates:
            gain = 0
            for t_sub in get_block_t_subs(block):
                if cover_count[t_sub_to_idx[t_sub]] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        solution_blocks.append(best_block)
        
        for t_sub in get_block_t_subs(best_block):
            idx = t_sub_to_idx[t_sub]
            if cover_count[idx] == 0:
                uncovered_count -= 1
            cover_count[idx] += 1

    # 4. Refinement: Remove redundant blocks
    # We need to be careful with cover_count updates
    for _ in range(5): # A few passes
        if time.time() > deadline:
            break
        any_removed = False
        for i in range(len(solution_blocks) - 1, -1, -1):
            block = solution_blocks[i]
            # Decrement
            block_t_subs = get_block_t_subs(block)
            for t_sub in block_t_subs:
                cover_count[t_sub_to_idx[t_sub]] -= 1
            
            # Check
            is_redundant = True
            for t_sub in block_t_subs:
                if cover_count[t_sub_to_idx[t_sub]] == 0:
                    is_redundant = False
                    break
            
            if is_redundant:
                solution_blocks.pop(i)
                any_removed = True
            else:
                # Restore
                for t_sub in block_t_subs:
                    cover_count[t_sub_to_idx[t_sub]] += 1
        if not any_removed:
            break

    return solution_blocks
