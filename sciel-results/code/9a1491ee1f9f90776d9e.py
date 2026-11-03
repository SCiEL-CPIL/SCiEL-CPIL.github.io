import itertools
import numpy as np
import time
import random

def _precompute(v, k, t):
    """Precompute t-subset indices and block-to-t-subset mapping."""
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Local t-combinations within a k-set
    local_t_combos = list(itertools.combinations(range(k), t))
    n_t_in_k = len(local_t_combos)
    
    # Build block_t_indices: (n_k, n_t_in_k)
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    for j, lc in enumerate(local_t_combos):
        lc_list = list(lc)
        for i, block in enumerate(k_subsets):
            t_combo = tuple(block[p] for p in lc_list)
            block_t_indices[i, j] = t_sub_to_idx[t_combo]
    
    # Build inverted index: for each t-subset, list of blocks containing it
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
            
    return t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks

def _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, seed=0):
    """Greedy set cover with incremental gain updates using inverted index."""
    rng = random.Random(seed)
    
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []
    
    # gains[i] = number of uncovered t-subsets in block i
    gains = np.full(n_k, n_t_in_k, dtype=np.int32)
    
    while True:
        if time.time() - start_time > deadline - 10.0:
            break
            
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with max gain, pick randomly among them
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        
        selected_blocks.append(best_idx)
        
        # Update uncovered and gains incrementally
        block_ts = block_t_indices[best_idx]
        for t_idx in block_ts:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                # Decrement gain for all blocks containing this t-subset
                for b in t_to_blocks[t_idx]:
                    gains[b] -= 1
        
    return selected_blocks

def _local_search_reduce(selected_block_indices, block_t_indices, n_t, n_k, t_to_blocks, start_time, deadline, seed=0):
    """
    Local search to reduce the number of blocks.
    1. Remove redundant blocks.
    2. If stuck, perform a 'rotation': remove a block, find a replacement that covers the lost subsets, 
       which may allow a different block to become redundant.
    """
    if not selected_block_indices:
        return selected_block_indices
    
    rng = random.Random(seed)
    selected_set = set(selected_block_indices)
    remaining = list(selected_block_indices)
    
    # Initial coverage count
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for bi in remaining:
        coverage_count[block_t_indices[bi]] += 1
    
    no_improvement_count = 0
    max_no_improvement = 500 # Cap on stuck iterations per restart
    
    while time.time() - start_time < deadline - 15.0:
        if no_improvement_count > max_no_improvement:
            break
            
        changed = False
        
        # Phase 1: Remove redundant blocks
        for i in range(len(remaining)):
            if time.time() - start_time > deadline - 15.0:
                return remaining
            bi = remaining[i]
            ts = block_t_indices[bi]
            if np.all(coverage_count[ts] > 1):
                # Block is redundant
                coverage_count[ts] -= 1
                remaining.pop(i)
                selected_set.discard(bi)
                changed = True
                no_improvement_count = 0
                break # Restart removal loop
        
        if changed:
            continue
            
        # Phase 2: Rotation / Swap
        # Pick a random block to remove temporarily
        if len(remaining) < 2:
            break
            
        idx = rng.choice(range(len(remaining)))
        bi_to_remove = remaining[idx]
        ts_to_remove = block_t_indices[bi_to_remove]
        
        # Find which t-subsets will become uncovered
        # A t-subset becomes uncovered if coverage_count[t] == 1
        # We need to find a new block that covers ALL such t-subsets.
        lost_ts = ts_to_remove[coverage_count[ts_to_remove] == 1]
        
        if len(lost_ts) == 0:
            # This block was redundant, but we should have caught it in Phase 1?
            # Maybe coverage counts changed? No, Phase 1 runs first.
            # If lost_ts is empty, it means the block doesn't uniquely cover anything.
            # Remove it.
            coverage_count[ts_to_remove] -= 1
            remaining.pop(idx)
            selected_set.discard(bi_to_remove)
            changed = True
            no_improvement_count = 0
            continue
            
        # Find a block that covers all lost_ts
        # Optimization: The new block must contain the intersection of candidates for each lost_ts
        # Or simpler: Iterate through candidates for the first lost_ts and check if they cover all others.
        
        # Get candidate blocks that cover the first lost t-subset
        first_lost = lost_ts[0]
        candidates = t_to_blocks[first_lost]
        
        replacement_found = False
        # Shuffle candidates to avoid bias
        rng.shuffle(candidates)
        
        for cand in candidates:
            if cand in selected_set:
                continue
            # Check if cand covers all lost_ts
            cand_ts = block_t_indices[cand]
            # We need to check if lost_ts is a subset of cand_ts
            # Since lost_ts is small (usually), we can check by verifying if each lost_t is in cand_ts
            # But checking membership in a numpy array is slow.
            # Better: Precompute a set? No, too slow per iteration.
            # Alternative: Use the fact that if cand covers first_lost, it's a candidate.
            # We can check if the intersection of (cand_ts) and (lost_ts) has size len(lost_ts).
            # Or simply: for lt in lost_ts: if lt not in cand_ts: break.
            # To speed up, we can sort lost_ts and cand_ts? No.
            # Let's just check.
            
            # Optimization: If the number of candidates is huge, this is slow.
            # But lost_ts usually restricts candidates significantly.
            
            is_valid = True
            for lt in lost_ts:
                # Check if lt is in cand_ts
                # Using np.isin is slow for single elements in large arrays repeatedly.
                # Let's use a boolean mask approach or just a set lookup if we precomputed sets?
                # We don't have sets for blocks.
                # We can use np.any(cand_ts == lt).
                if not np.any(cand_ts == lt):
                    is_valid = False
                    break
                    
            if is_valid:
                # Perform swap
                # Remove old
                coverage_count[ts_to_remove] -= 1
                remaining.pop(idx)
                selected_set.discard(bi_to_remove)
                
                # Add new
                remaining.append(cand)
                selected_set.add(cand)
                coverage_count[block_t_indices[cand]] += 1
                
                replacement_found = True
                no_improvement_count = 0
                break
        
        if not replacement_found:
            no_improvement_count += 1

    return remaining

def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    
    # Precompute
    t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks = _precompute(v, k, t)
    
    # Multiple restarts with different seeds
    best_result = None
    best_count = float('inf')
    
    seed = 0
    max_restarts = 50
    while seed < max_restarts:
        if time.time() - start_time > deadline - 20.0:
            break
        
        # 1. Greedy Cover
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, seed)
        
        # 2. Local Search to Reduce
        selected = _local_search_reduce(selected, block_t_indices, n_t, n_k, t_to_blocks, start_time, deadline, seed)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_result = list(selected)
        
        # Check time for next restart
        if time.time() - start_time > deadline - 25.0:
            break
        
        seed += 1
    
    if best_result is None:
        # Fallback
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, 0)
        selected = _local_search_reduce(selected, block_t_indices, n_t, n_k, t_to_blocks, start_time, deadline, 0)
        best_result = selected
    
    return [list(k_subsets[bi]) for bi in best_result]
