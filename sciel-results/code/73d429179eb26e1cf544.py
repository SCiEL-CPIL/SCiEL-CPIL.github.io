"""Covering design C(v,k,t) via bitmask greedy + aggressive local search with restarts."""
import itertools
import time
import random


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin under 120s limit

    def _deadline_ok(margin=5.0):
        return time.time() < deadline - margin

    # 1. Precomputation
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # Precompute for each block: list of t-subset indices it covers
    block_subsets = []
    block_masks = [0] * num_blocks
    for i in range(num_blocks):
        block = all_blocks[i]
        subs = []
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = sub_index[ts]
            subs.append(idx)
            mask |= (1 << idx)
        block_subsets.append(subs)
        block_masks[i] = mask

    # Precompute tset_to_blocks: for each t-subset, which blocks cover it
    tset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_subsets[b_idx]:
            tset_to_blocks[s_idx].append(b_idx)

    all_mask = (1 << num_subsets) - 1

    # 2. Greedy Set Cover with randomization
    def _greedy_cover(rng):
        covered = 0
        selected = []
        selected_set = set()
        
        # Shuffle block order for diversity
        block_order = list(range(num_blocks))
        rng.shuffle(block_order)
        
        while covered != all_mask:
            if not _deadline_ok(10.0):
                break
            
            best_block_idx = -1
            best_gain = 0
            
            for i in block_order:
                if i in selected_set:
                    continue
                gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1 or best_gain == 0:
                break
            
            selected.append(best_block_idx)
            selected_set.add(best_block_idx)
            covered |= block_masks[best_block_idx]
        
        return selected, selected_set, covered

    # 3. Local Search
    def _local_search(selected, selected_set, covered, rng, time_budget):
        # Initialize counts
        subset_counts = [0] * num_subsets
        for idx in selected:
            for s_idx in block_subsets[idx]:
                subset_counts[s_idx] += 1
        
        best_sol = list(selected)
        best_size = len(selected)
        
        no_improve = 0
        max_no_improve = 30
        
        while _deadline_ok(5.0) and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Deletion - remove redundant blocks
            while _deadline_ok(5.0):
                redundant_idx = -1
                for idx in list(selected_set):
                    redundant = True
                    for s_idx in block_subsets[idx]:
                        if subset_counts[s_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        redundant_idx = idx
                        break
                
                if redundant_idx == -1:
                    break
                
                selected_set.remove(redundant_idx)
                for s_idx in block_subsets[redundant_idx]:
                    subset_counts[s_idx] -= 1
                selected.remove(redundant_idx)
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = list(selected)
            
            if not _deadline_ok(5.0):
                break
            
            # Phase 2: Swap - try to replace a block to enable future deletions
            if selected:
                found_swap = False
                max_swap_attempts = 15
                if _deadline_ok(30.0):
                    max_swap_attempts = 30
                
                for _ in range(max_swap_attempts):
                    if not _deadline_ok(5.0):
                        break
                    if not selected:
                        break
                    
                    # Pick a random block from the solution
                    pos = rng.randint(0, len(selected) - 1)
                    b_out = selected[pos]
                    
                    # Find critical subsets (count == 1)
                    critical = []
                    for s_idx in block_subsets[b_out]:
                        if subset_counts[s_idx] == 1:
                            critical.append(s_idx)
                    
                    if not critical:
                        # Redundant - should have been caught
                        continue
                    
                    # Pick the critical subset with fewest covering blocks
                    min_ts = critical[0]
                    min_len = len(tset_to_blocks[min_ts])
                    for ts in critical[1:]:
                        curr_len = len(tset_to_blocks[ts])
                        if curr_len < min_len:
                            min_ts = ts
                            min_len = curr_len
                    
                    # Build critical mask for fast checking
                    critical_mask = 0
                    for ts in critical:
                        critical_mask |= (1 << ts)
                    
                    # Find candidates from the shortest list
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts]:
                        if b_idx in selected_set:
                            continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)
                    
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        
                        # Perform swap
                        selected_set.remove(b_out)
                        for s in block_subsets[b_out]:
                            subset_counts[s] -= 1
                        
                        selected[pos] = b_in
                        selected_set.add(b_in)
                        for s in block_subsets[b_in]:
                            subset_counts[s] += 1
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    no_improve = 0
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_sol = list(selected)
                    continue
            
            no_improve += 1
        
        return best_sol

    # Main: Try multiple greedy restarts with local search
    best_solution = None
    best_size = float('inf')
    
    # Determine number of restarts based on time
    # First greedy + local search, then use remaining time for more
    rng = random.Random(42)
    
    # Initial greedy
    selected, selected_set, covered = _greedy_cover(rng)
    
    if not _deadline_ok(10.0):
        return [list(all_blocks[i]) for i in selected]
    
    # Local search on first solution
    result = _local_search(selected, selected_set, covered, rng, 60.0)
    if result:
        best_solution = result
        best_size = len(result)
    
    # Try additional restarts if time permits
    restart_count = 0
    while _deadline_ok(15.0) and restart_count < 10:
        restart_count += 1
        rng2 = random.Random(42 + restart_count)
        selected, selected_set, covered = _greedy_cover(rng2)
        
        if not _deadline_ok(10.0):
            break
        
        result = _local_search(selected, selected_set, covered, rng2, 20.0)
        if result and len(result) < best_size:
            best_solution = result
            best_size = len(result)
    
    if best_solution is None:
        return [list(all_blocks[i]) for i in selected]
    
    return [list(all_blocks[i]) for i in best_solution]
