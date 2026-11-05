import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin for safety
    
    # 1. Precompute all t-subsets and their indices
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Build incidence structure: for each block, list of t-subset indices it covers
    # Using arrays for fast numpy operations
    block_to_subsets = []
    for block in all_blocks:
        subs = np.array([subset_to_idx[s] for s in itertools.combinations(block, t)], dtype=np.int32)
        block_to_subsets.append(subs)
    
    # 4. Build reverse incidence: for each t-subset, list of blocks that cover it
    # This is important for gain updates
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for u in subs:
            subset_to_blocks[u].append(b_idx)
    
    # Convert to arrays for speed
    subset_to_blocks_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in subset_to_blocks]
    
    # 5. Greedy set cover with dynamic gains
    uncovered = np.ones(num_subsets, dtype=bool)
    gains = np.zeros(num_blocks, dtype=np.int32)
    
    # Initial gains: for each uncovered subset, increment gain of blocks covering it
    for u in range(num_subsets):
        if uncovered[u]:
            blocks_covering = subset_to_blocks_arr[u]
            if len(blocks_covering) > 0:
                gains[blocks_covering] += 1
    
    blocks_found = []
    block_indices_found = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Find block with maximum gain
        b_star = int(np.argmax(gains))
        if gains[b_star] <= 0:
            break  # No block can improve coverage
        
        # Add this block
        blocks_found.append(list(all_blocks[b_star]))
        block_indices_found.append(b_star)
        
        # Update coverage: mark newly covered subsets
        block_subs = block_to_subsets[b_star]
        newly_covered = block_subs[uncovered[block_subs]]
        
        if len(newly_covered) > 0:
            uncovered[newly_covered] = False
            # Update gains: decrement for all blocks that also cover these subsets
            for u in newly_covered:
                blocks_covering = subset_to_blocks_arr[u]
                if len(blocks_covering) > 0:
                    gains[blocks_covering] -= 1
    
    # 6. Pruning phase: remove redundant blocks
    remaining_time = deadline - time.time()
    if remaining_time > 20 and len(blocks_found) > 1:
        # Build coverage count for each t-subset
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in block_indices_found:
            subs = block_to_subsets[b_idx]
            cover_count[subs] += 1
        
        # Try to remove blocks in order of lowest "essentiality"
        # A block is removable if all its subsets are covered by at least 2 blocks
        # Iterate multiple passes
        for pass_num in range(10):
            if time.time() > deadline - 10:
                break
            
            removed_this_pass = False
            # Sort blocks by number of subsets they exclusively cover (ascending)
            # For efficiency, just check in reverse order (last added first)
            for i in range(len(block_indices_found) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                b_idx = block_indices_found[i]
                subs = block_to_subsets[b_idx]
                
                # Check if all subsets in this block are covered by >= 2 blocks
                # (i.e., even after removing this block, they're still covered)
                if len(subs) > 0 and np.min(cover_count[subs]) >= 2:
                    # This block is redundant
                    cover_count[subs] -= 1
                    block_indices_found.pop(i)
                    blocks_found.pop(i)
                    removed_this_pass = True
            
            if not removed_this_pass:
                break
    
    return blocks_found
