import itertools
import time
import sys
import numpy as np

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
    
    # Precompute block_to_t_subsets as sets for fast membership testing
    block_to_t_subsets_set = [set(ts) for ts in block_to_t_subsets]
    
    # 3. Greedy Construction with NumPy acceleration
    # Use bitmask representation: each t-subset gets a bit position
    # For n_t up to ~100k, we need arrays of uint64
    n_bits_per_block = (n_t + 63) // 64
    # Coverage matrix: n_k x n_bits_per_block, where each entry is a uint64 bitmask
    # But this might be too large. Let's use a different approach.
    
    # Actually, let's use a sparse approach with NumPy arrays for gains
    # Maintain an array of which t-subsets are uncovered
    uncovered_mask = np.ones(n_t, dtype=np.int32)  # 1 if uncovered, 0 if covered
    
    active_blocks = []
    
    # Precompute for each block, the list of t-subset indices it covers
    # We already have block_to_t_subsets as lists
    
    while np.any(uncovered_mask):
        if time.time() > deadline:
            break
            
        best_block = -1
        best_gain = -1
        
        # Vectorized gain computation
        # For each block, sum uncovered_mask over its covered t-subsets
        # This is tricky to vectorize directly. Let's use a hybrid approach.
        
        # Compute gains for all blocks
        gains = np.zeros(n_k, dtype=np.int32)
        for b_idx, covered_ts in enumerate(block_to_t_subsets):
            if covered_ts:
                gains[b_idx] = uncovered_mask[covered_ts].sum()
        
        best_block = int(np.argmax(gains))
        best_gain = int(gains[best_block])
        
        if best_gain <= 0:
            break  # Should not happen if uncovered is not empty
            
        active_blocks.append(best_block)
        
        # Update uncovered mask
        for s in block_to_t_subsets[best_block]:
            uncovered_mask[s] = 0
                
    # 4. Local Search: Redundancy Removal and Swaps
    # We need cover_count to efficiently check redundancy and swaps.
    # Rebuild cover_count from active_blocks
    cover_count = np.zeros(n_t, dtype=np.int32)
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
        swapped_any = False
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
                
            # Try to find a swap: find a non-active block that covers all unique_ts
            found_swap = False
            for b_new in range(n_k):
                if b_new in active_set:
                    continue
                    
                # Check if b_new covers all unique_ts
                covers_all = all(s in block_to_t_subsets_set[b_new] for s in unique_ts)
                
                if covers_all:
                    # Perform the swap
                    # Remove b_idx
                    for s in block_to_t_subsets[b_idx]:
                        cover_count[s] -= 1
                    # Add b_new
                    for s in block_to_t_subsets[b_new]:
                        cover_count[s] += 1
                    active_set.remove(b_idx)
                    active_set.add(b_new)
                    found_swap = True
                    swapped_any = True
                    improved = True
                    break
            
            if not found_swap:
                new_active_blocks.append(b_idx)
            else:
                new_active_blocks.append(b_new)
            
        active_blocks = new_active_blocks
        active_set = set(active_blocks)
        
        if not swapped_any:
            break
    
    # Convert block indices back to lists of points
    result = [list(all_k_blocks[b_idx]) for b_idx in active_blocks]
    
    # Ensure distinct blocks
    seen = set()
    final_result = []
    for b in result:
        b_tuple = tuple(b)
        if b_tuple not in seen:
            seen.add(b_tuple)
            final_result.append(b)
            
    return final_result
