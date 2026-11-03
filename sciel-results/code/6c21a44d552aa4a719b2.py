def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    
    block_covers = []
    for b in all_k_blocks:
        b_set = set(b)
        covered = []
        for s in itertools.combinations(b, t):
            idx = t_subset_to_idx[s]
            covered.append(idx)
        block_covers.append(covered)
        
    # Greedy
    uncovered = [True] * num_t_subsets
    block_counts = [len(covers) for covers in block_covers]
    blocks = []
    
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, covers in enumerate(block_covers):
        for s_idx in covers:
            t_subset_to_blocks[s_idx].append(b_idx)
            
    start_time = time.time()
    greedy_time_limit = 10.0
    
    while any(uncovered):
        if time.time() - start_time > greedy_time_limit:
            break
        best_block = -1
        max_count = -1
        for b_idx in range(len(all_k_blocks)):
            if block_counts[b_idx] > max_count:
                max_count = block_counts[b_idx]
                best_block = b_idx
        if best_block == -1: break
        blocks.append(all_k_blocks[best_block])
        for s_idx in block_covers[best_block]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                for b_idx in t_subset_to_blocks[s_idx]:
                    block_counts[b_idx] -= 1
                    
    initial_blocks = blocks
    
    # Local Search
    current_state = [all_k_blocks.index(b) for b in initial_blocks]
    
    cover_counts = [0] * num_t_subsets
    for b_idx in current_state:
        for s_idx in block_covers[b_idx]:
            cover_counts[s_idx] += 1
            
    uncovered_count = sum(1 for c in cover_counts if c == 0)
    
    # If not valid, add random blocks
    while uncovered_count > 0:
        b_idx = random.randint(0, len(all_k_blocks) - 1)
        current_state.append(b_idx)
        for s_idx in block_covers[b_idx]:
            if cover_counts[s_idx] == 0:
                uncovered_count -= 1
            cover_counts[s_idx] += 1
            
    best_state = current_state.copy()
    best_uncovered = 0
    
    start_ls_time = time.time()
    ls_time_limit = 120.0 - (time.time() - start_time) - 5.0
    
    while time.time() - start_ls_time > 0:
        # Try Remove
        if len(current_state) > 1:
            idx = random.randint(0, len(current_state) - 1)
            b_idx = current_state[idx]
            
            # Tentative remove
            new_uncovered = 0
            for s_idx in block_covers[b_idx]:
                if cover_counts[s_idx] == 1:
                    new_uncovered += 1
                cover_counts[s_idx] -= 1
                
            if new_uncovered == 0:
                # Accept
                current_state.pop(idx)
                if len(current_state) < len(best_state):
                    best_state = current_state.copy()
            else:
                # Revert
                for s_idx in block_covers[b_idx]:
                    cover_counts[s_idx] += 1
                    
        # Try Swap
        if len(current_state) > 0:
            idx = random.randint(0, len(current_state) - 1)
            b_idx_old = current_state[idx]
            b_idx_new = random.randint(0, len(all_k_blocks) - 1)
            
            if b_idx_new != b_idx_old:
                # Tentative swap
                # Remove old
                new_uncovered = 0
                for s_idx in block_covers[b_idx_old]:
                    if cover_counts[s_idx] == 1:
                        new_uncovered += 1
                    cover_counts[s_idx] -= 1
                    
                # Add new
                for s_idx in block_covers[b_idx_new]:
                    if cover_counts[s_idx] == 0:
                        new_uncovered -= 1
                    cover_counts[s_idx] += 1
                    
                if new_uncovered == 0:
                    # Accept
                    current_state[idx] = b_idx_new
                else:
                    # Revert
                    for s_idx in block_covers[b_idx_new]:
                        cover_counts[s_idx] -= 1
                    for s_idx in block_covers[b_idx_old]:
                        cover_counts[s_idx] += 1

    best_blocks = [all_k_blocks[i] for i in best_state]
    return best_blocks
