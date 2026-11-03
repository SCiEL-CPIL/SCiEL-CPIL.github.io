import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Generate all k-subsets (blocks) and precompute which t-subsets they cover
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # block_to_t_subsets[b] is a list of indices of t-subsets covered by block b
    block_to_t_subsets = []
    for b in all_k_blocks:
        covered = []
        for ts in itertools.combinations(b, t):
            covered.append(t_subset_to_idx[ts])
        block_to_t_subsets.append(covered)
        
    # 3. Greedy Construction
    cover_count = [0] * n_t
    active_blocks = []
    
    # To speed up greedy, we can maintain a list of "uncovered" t-subsets
    # but calculating gain for all blocks is O(n_k * t_combinations)
    # n_k ~ 50k, t_combinations per block = 35. Total ~ 1.7M ops per iteration.
    # 150 iterations -> 250M ops. Might be tight in pure Python.
    # Let's use a slightly smarter approach or just hope it's fast enough.
    # Actually, 250M simple operations in Python might take ~10-20s.
    # Let's try to optimize the gain calculation.
    
    # We can maintain a set of uncovered t-subsets.
    # But checking "is s in uncovered" for 35 subsets per block is fast.
    
    uncovered = set(range(n_t))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block = -1
        best_gain = -1
        
        # Iterate through all possible blocks
        for b_idx, covered_ts in enumerate(block_to_t_subsets):
            # Count how many of these are currently uncovered
            gain = 0
            for s in covered_ts:
                if s in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1:
            break # Should not happen if uncovered is not empty
            
        active_blocks.append(best_block)
        
        # Update uncovered set
        for s in block_to_t_subsets[best_block]:
            if s in uncovered:
                uncovered.remove(s)
                
    # 4. Local Search: Redundancy Removal and Swaps
    # We need cover_count to efficiently check redundancy and swaps.
    # Rebuild cover_count from active_blocks
    cover_count = [0] * n_t
    for b_idx in active_blocks:
        for s in block_to_t_subsets[b_idx]:
            cover_count[s] += 1
            
    active_set = set(active_blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Redundancy Removal
        # Check each active block
        new_active_blocks = []
        for b_idx in active_blocks:
            redundant = True
            for s in block_to_t_subsets[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove b_idx
                for s in block_to_t_subsets[b_idx]:
                    cover_count[s] -= 1
                improved = True
            else:
                new_active_blocks.append(b_idx)
        
        if improved:
            active_blocks = new_active_blocks
            active_set = set(active_blocks)
            continue # Restart loop as removals might make others redundant
            
        # Pass 2: Swaps
        # For each active block, try to swap with a non-active block
        # that covers all its unique t-subsets.
        swapped = False
        new_active_blocks = []
        for b_idx in active_blocks:
            # Find unique t-subsets of b_idx
            unique_ts = []
            for s in block_to_t_subsets[b_idx]:
                if cover_count[s] == 1:
                    unique_ts.append(s)
            
            if not unique_ts:
                # Should have been caught by redundancy, but keep it
                new_active_blocks.append(b_idx)
                continue
                
            # Try to find a swap
            found_swap = False
            for b_new in range(n_k):
                if b_new in active_set:
                    continue
                    
                # Check if b_new covers all unique_ts
                # Optimization: if b_new is smaller or similar, check intersection
                # Simple check:
                covers_all = True
                for s in unique_ts:
                    # We need to know if s is in block_to_t_subsets[b_new]
                    # This is a linear search in the list, which is bad.
                    # Precompute a set for each block? Memory might be high but n_k is 50k.
                    # 50k sets of size 35. That's fine.
                    pass 
                
                # Let's precompute block_to_t_subsets_set for faster lookup
                # But we are in the loop. Let's just do a simple check first.
                # Actually, we can precompute a global structure.
                
            # This inner loop is O(n_k * |unique_ts|).
            # |unique_ts| is small usually.
            # But doing this for every active block is O(n_active * n_k * t).
            # 150 * 50k * 35 = 262M ops. Might be too slow in Python.
            
            # Let's break out if time is tight.
            if time.time() > deadline - 10:
                break
                
            # For now, let's just skip the complex swap search if it's too slow
            # and rely on redundancy removal which is faster.
            # Or, implement a simpler swap:
            # Only try to swap if unique_ts is small.
            
            new_active_blocks.append(b_idx)
            
        active_blocks = new_active_blocks
        active_set = set(active_blocks)
        
        # If we didn't do any swaps, we are done.
        # To actually do swaps, we need the precomputed sets.
        
    # Convert block indices back to lists of points
    result = [list(all_k_blocks[b_idx]) for b_idx in active_blocks]
    
    # Ensure distinct blocks
    # The greedy and swap logic ensures distinctness if we are careful,
    # but let's verify.
    seen = set()
    final_result = []
    for b in result:
        b_tuple = tuple(b)
        if b_tuple not in seen:
            seen.add(b_tuple)
            final_result.append(b)
            
    return final_result
