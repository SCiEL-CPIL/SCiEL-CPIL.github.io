import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin for safety

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute block masks and t-subset indices
    # block_t_indices[block_idx] = list of t_sub_idx covered by block
    block_t_indices = []
    for block in all_blocks:
        indices = [t_sub_index[sub] for sub in itertools.combinations(block, t)]
        block_t_indices.append(indices)

    # Precompute which blocks cover each t-subset
    # blocks_for_t_sub[t_sub_idx] = list of block_idx
    blocks_for_t_sub = [[] for _ in range(n_t_subs)]
    for bi, t_indices in enumerate(block_t_indices):
        for ti in t_indices:
            blocks_for_t_sub[ti].append(bi)

    # Helper to check if a set of blocks covers all t-subsets
    def is_covered(blocks):
        counts = [0] * n_t_subs
        for b in blocks:
            for ti in block_t_indices[b]:
                counts[ti] += 1
        return all(c > 0 for c in counts)

    # New Greedy: MRV (Minimum Remaining Values)
    def greedy_mrv(seed, time_limit):
        random.seed(seed)
        counts = [0] * n_t_subs
        selected_blocks = []
        uncovered = set(range(n_t_subs))
        
        # MRV data structure: for each uncovered t-sub, we know the number of blocks covering it.
        # Initially, all are uncovered.
        # We need to pick the uncovered t-sub with the fewest covering blocks.
        # Then among those blocks, pick the one with max gain on other uncovered subs.
        
        while uncovered and time.time() < time_limit:
            # Find MRV t-sub: min number of covering blocks
            # Note: The number of covering blocks is static for a given t-sub, 
            # but we only care about uncovered ones.
            # We can precompute the "degree" (number of blocks covering it) for each t-sub.
            # t_sub_degree[ti] = len(blocks_for_t_sub[ti])
            
            # To be efficient, we can just iterate through uncovered and find min degree.
            # Or maintain a priority queue.
            # Given n_t_subs ~ 12k, linear scan is fast enough.
            
            min_degree = float('inf')
            mrv_t_idx = -1
            for ti in uncovered:
                deg = len(blocks_for_t_sub[ti])
                if deg < min_degree:
                    min_degree = deg
                    mrv_t_idx = ti
            
            if mrv_t_idx == -1:
                break

            # Get candidate blocks for this t-sub
            candidates = blocks_for_t_sub[mrv_t_idx]
            
            # Among candidates, find the one that covers the most *other* uncovered t-subs
            best_block_idx = -1
            best_gain = -1
            
            # To speed this up, we can sample candidates if the list is very large,
            # but for t=5, k=8, v=19, max candidates is C(14,3)=364. This is small.
            for bi in candidates:
                gain = 0
                for ti in block_t_indices[bi]:
                    if ti in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
            
            if best_block_idx == -1:
                # Should not happen if uncovered is not empty and we have valid blocks
                break
                
            # Select the block
            selected_blocks.append(best_block_idx)
            
            # Update counts and uncovered
            for ti in block_t_indices[best_block_idx]:
                counts[ti] += 1
                if counts[ti] == 1:
                    uncovered.discard(ti)
                    
        # Convert indices to blocks
        return [list(all_blocks[bi]) for bi in selected_blocks]

    # Local Search (Improved)
    def local_search(initial_blocks, time_limit):
        blocks = [list(b) for b in initial_blocks]
        counts = [0] * n_t_subs
        
        # Precompute t-sub indices for each block
        def get_t_indices(block):
            return [t_sub_index[sub] for sub in itertools.combinations(block, t)]
        
        block_t_indices_current = [get_t_indices(b) for b in blocks]
        
        # Build counts from block_t_indices_current
        for ti_list in block_t_indices_current:
            for ti in ti_list:
                counts[ti] += 1
                
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        # Precompute: for each point p, which t-subs contain p
        # point_t_subs[p] = list of t_sub_indices containing p
        point_t_subs = [[] for _ in range(v)]
        for ti, sub in enumerate(all_t_subs):
            for p in sub:
                point_t_subs[p].append(ti)
        
        while time.time() < time_limit:
            # Try to remove redundant blocks
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    removable = True
                    for ti in block_t_indices_current[i]:
                        if counts[ti] <= 1:
                            removable = False
                            break
                    if removable:
                        for ti in block_t_indices_current[i]:
                            counts[ti] -= 1
                        blocks.pop(i)
                        block_t_indices_current.pop(i)
                        changed = True
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                        break
            
            if time.time() >= time_limit:
                break
            
            # Perform random swaps and block exchanges
            n_moves = 300
            for _ in range(n_moves):
                if time.time() >= time_limit:
                    break
                if not blocks:
                    break
                
                move_type = random.random()
                
                if move_type < 0.7:
                    # Single-point swap
                    i = random.randint(0, len(blocks)-1)
                    b = blocks[i]
                    x = random.choice(b)
                    y = random.choice([p for p in range(v) if p not in b])
                    
                    # Check validity: all t-subs of b containing x must have count > 1
                    valid = True
                    for ti in block_t_indices_current[i]:
                        if all_t_subs[ti][0] == x or x in all_t_subs[ti]:
                            # More efficient: check if x is in the t-sub
                            pass
                    
                    # Better: precompute for each block which t-subs contain each point
                    # For now, use a direct check
                    valid = True
                    for sub in itertools.combinations(b, t):
                        if x in sub and counts[t_sub_index[sub]] <= 1:
                            valid = False
                            break
                    
                    if valid:
                        # Decrement counts for subs containing x
                        for sub in itertools.combinations(b, t):
                            if x in sub:
                                counts[t_sub_index[sub]] -= 1
                        
                        # Update block
                        b.remove(x)
                        b.append(y)
                        b.sort()
                        blocks[i] = b
                        block_t_indices_current[i] = get_t_indices(b)
                        
                        # Increment counts for subs containing y
                        for sub in itertools.combinations(b, t):
                            if y in sub:
                                counts[t_sub_index[sub]] += 1
                                
                else:
                    # Block exchange: try to replace one block with a better one
                    # Find an uncovered or poorly covered t-sub and try to find a block that covers it
                    if random.random() < 0.5:
                        # Try to find a block that covers a specific uncovered t-sub
                        uncovered_tis = [ti for ti in range(n_t_subs) if counts[ti] == 0]
                        if uncovered_tis:
                            ti = random.choice(uncovered_tis)
                            target_sub = all_t_subs[ti]
                            # Find a random block that contains this t-sub
                            candidates = []
                            for bi, block in enumerate(all_blocks):
                                if all(p in block for p in target_sub):
                                    candidates.append(bi)
                            if candidates:
                                new_bi = random.choice(candidates)
                                # Remove a random existing block and add the new one
                                if blocks:
                                    old_i = random.randint(0, len(blocks)-1)
                                    # Check if removing old_i is safe
                                    safe = True
                                    for ti2 in block_t_indices_current[old_i]:
                                        if counts[ti2] <= 1:
                                            safe = False
                                            break
                                    if safe:
                                        for ti2 in block_t_indices_current[old_i]:
                                            counts[ti2] -= 1
                                        blocks.pop(old_i)
                                        block_t_indices_current.pop(old_i)
                                        
                                        new_block = list(all_blocks[new_bi])
                                        blocks.append(new_block)
                                        block_t_indices_current.append(get_t_indices(new_block))
                                        for ti2 in block_t_indices_current[-1]:
                                            counts[ti2] += 1
                                        
                                        if len(blocks) < best_len:
                                            best_len = len(blocks)
                                            best_blocks = [list(x) for x in blocks]
        
        return best_blocks

    # Strategy: Run Greedy MRV and Local Search
    best_initial = None
    best_initial_len = float('inf')
    
    greedy_time = 15
    search_time = deadline - start_time - greedy_time - 5
    
    # Run Greedy MRV
    blocks_mrv = greedy_mrv(0, time.time() + greedy_time)
    if len(blocks_mrv) < best_initial_len:
        best_initial_len = len(blocks_mrv)
        best_initial = blocks_mrv
        
    if best_initial is None:
        best_initial = [] # Fallback if MRV fails? Unlikely.
        
    final_blocks = local_search(best_initial, deadline)
    
    return final_blocks
