import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and k-subsets
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    all_k_subsets = list(itertools.combinations(points, k))
    
    num_t = len(all_t_subsets)
    num_k = len(all_k_subsets)
    
    # Map t-subset to index
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each k-block, the list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_to_idx[ts])
        block_t_indices.append(indices)
        
    # Precompute for each t-subset, the list of k-block indices that cover it
    # This is useful for finding candidates for swap
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_to_blocks[idx].append(b_idx)

    def greedy_set_cover():
        uncovered = [True] * num_t
        # gain[b] = number of currently uncovered t-subsets covered by block b
        gain = [len(block_t_indices[b]) for b in range(num_k)]
        selected = []
        
        while True:
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = [i for i, g in enumerate(gain) if g == max_gain]
            best_block = random.choice(candidates)
            selected.append(best_block)
            
            # Update uncovered and gains
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # This t_idx is now covered, so it no longer contributes to gain
                    # for any block covering it
                    for b_idx in t_to_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected

    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            # Iterate over a copy to allow deletion
            for b_idx in list(active_blocks_set):
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    def local_search(active_blocks_set, cover_counts, time_limit):
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            current_blocks = list(active_blocks_set)
            random.shuffle(current_blocks)
            
            for B in current_blocks:
                if B not in active_blocks_set:
                    continue
                
                # Find t-subsets uniquely covered by B
                unique_ts = []
                for t_idx in block_t_indices[B]:
                    if cover_counts[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if not unique_ts:
                    # B is redundant? Minimize would have caught it, 
                    # but maybe it wasn't run recently.
                    # If no unique ts, we can try to swap with any block not in set?
                    # No, if no unique ts, removing B is valid. 
                    # But minimize handles removal.
                    # For swap, we need to replace B with B'.
                    # If B has no unique coverage, we can swap with ANY B' 
                    # that doesn't break coverage? 
                    # Actually, if we remove B, coverage is still full.
                    # So we can just remove B.
                    # But we are in local_search, minimize is called separately.
                    # Let's just skip swap if no unique ts, minimize will handle it.
                    continue

                # Find candidates B' that cover all unique_ts
                # Intersect the sets of blocks covering each unique t
                candidates = None
                for t_idx in unique_ts:
                    blocks_list = t_to_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Remove B itself from candidates
                candidates.discard(B)
                
                # We want to swap B with B'.
                # Ideally, B' should add new coverage or at least not reduce "flexibility".
                # But primarily, we just need a valid swap.
                # To improve, we might prefer B' that has more "redundancy" or 
                # allows future removals. 
                # For now, random choice from valid candidates.
                if candidates:
                    B_new = random.choice(list(candidates))
                    
                    # Perform Swap
                    active_blocks_set.remove(B)
                    active_blocks_set.add(B_new)
                    
                    for t_idx in block_t_indices[B]:
                        cover_counts[t_idx] -= 1
                    for t_idx in block_t_indices[B_new]:
                        cover_counts[t_idx] += 1
                    
                    # Minimize after swap
                    active_blocks_set, cover_counts = minimize(active_blocks_set, cover_counts)
                    improved = True
            
            if not improved:
                break
        
        return active_blocks_set, cover_counts

    best_blocks = None
    best_size = float('inf')
    total_time_limit = 115.0  # Safety margin
    start_total = time.time()
    
    # Run multiple iterations
    while time.time() - start_total < total_time_limit:
        # 1. Greedy
        selected = greedy_set_cover()
        active_blocks_set = set(selected)
        
        # 2. Calculate cover counts
        cover_counts = [0] * num_t
        for b_idx in active_blocks_set:
            for t_idx in block_t_indices[b_idx]:
                cover_counts[t_idx] += 1
        
        # 3. Minimize
        active_blocks_set, cover_counts = minimize(active_blocks_set, cover_counts)
        
        # 4. Local Search
        # Allocate time based on remaining
        elapsed = time.time() - start_total
        remaining = total_time_limit - elapsed
        if remaining < 5.0:
            break
        
        # Use a fraction of remaining time for local search
        # If we have lots of time, use more.
        ls_time = max(1.0, remaining * 0.2) # Use 20% of remaining time for LS
        active_blocks_set, cover_counts = local_search(active_blocks_set, cover_counts, ls_time)
        
        if len(active_blocks_set) < best_size:
            best_size = len(active_blocks_set)
            best_blocks = list(active_blocks_set)
            
            # If we hit a very good score, we might want to spend more time polishing?
            # But restarts are good for escaping local minima.
            
    if best_blocks is None:
        # Fallback
        selected = greedy_set_cover()
        active_blocks_set = set(selected)
        cover_counts = [0] * num_t
        for b_idx in active_blocks_set:
            for t_idx in block_t_indices[b_idx]:
                cover_counts[t_idx] += 1
        active_blocks_set, cover_counts = minimize(active_blocks_set, cover_counts)
        best_blocks = list(active_blocks_set)

    return [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
