import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Convert to numpy arrays for fast operations
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]

    # Initialize gain vector: gain[b] = number of currently uncovered subsets in block b
    uncovered = np.ones(n_subsets, dtype=bool)
    gains = np.zeros(n_blocks, dtype=np.int32)

    # Initial gains: for each subset, add 1 to all blocks containing it
    for u in range(n_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1

    blocks_found = []
    used_blocks = set()

    while True:
        if time.time() > deadline:
            break

        if not uncovered.any():
            break

        # Find block with max gain
        b_star = int(np.argmax(gains))

        if gains[b_star] <= 0:
            break

        # Add this block
        blocks_found.append(list(all_blocks[b_star]))
        used_blocks.add(b_star)

        # Update uncovered and gains
        for u in block_to_subsets[b_star]:
            if uncovered[u]:
                uncovered[u] = False
                # Decrement gain for all blocks containing u
                for b in subset_to_blocks[u]:
                    gains[b] -= 1

        # Early exit if all covered
        if not uncovered.any():
            break

    # Polish pass: try to improve by checking if we can replace some blocks
    # with better ones, or add targeted blocks for remaining uncovered
    if time.time() < deadline - 5.0 and uncovered.any():
        # For each uncovered subset, find the block with max gain among unused blocks
        # that covers it, and add it
        uncov_indices = np.where(uncovered)[0]
        for u_idx in uncov_indices:
            if time.time() > deadline - 2.0:
                break
            best_b = -1
            best_g = 0
            for b in subset_to_blocks[u_idx]:
                if b not in used_blocks:
                    g = gains[b]
                    if g > best_g:
                        best_g = g
                        best_b = b
            if best_b >= 0 and best_g > 0:
                blocks_found.append(list(all_blocks[best_b]))
                used_blocks.add(best_b)
                for u in block_to_subsets[best_b]:
                    if uncovered[u]:
                        uncovered[u] = False
                        for b in subset_to_blocks[u]:
                            gains[b] -= 1
                if not uncovered.any():
                    break

    # Local search: try to remove redundant blocks
    # For each block, check if removing it still leaves all subsets covered
    if time.time() < deadline - 3.0:
        # Build coverage count: for each subset, how many selected blocks cover it
        selected_set = set(range(len(blocks_found)))
        # Map from block index in all_blocks to its position in blocks_found
        # Actually blocks_found contains lists, not indices. Let's rebuild.
        # Rebuild: for each block in blocks_found, find its index in all_blocks
        block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks_found]
        
        # Count how many selected blocks cover each subset
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            for u in block_to_subsets[b_idx]:
                cover_count[u] += 1
        
        # Try to remove blocks that don't reduce coverage below 1
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Recompute cover_count from scratch each iteration is expensive
            # Instead, try removing each block and check if any subset drops to 0
            for i in range(len(selected_block_indices) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                b_idx = selected_block_indices[i]
                # Check if this block is the only one covering any subset
                redundant = True
                for u in block_to_subsets[b_idx]:
                    if cover_count[u] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    selected_block_indices.pop(i)
                    # Update cover_count
                    for u in block_to_subsets[b_idx]:
                        cover_count[u] -= 1
                    improved = True
                    break  # Restart the loop since indices changed
        
        # Rebuild blocks_found from selected_block_indices
        blocks_found = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]

    # Additional local search: try to swap blocks to reduce total count
    # Try removing one block and see if we can cover its unique subsets with one other block
    if time.time() < deadline - 5.0:
        # Rebuild structures for the current solution
        block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks_found]
        
        # Count how many selected blocks cover each subset
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            for u in block_to_subsets[b_idx]:
                cover_count[u] += 1
        
        # Try to find a block that can be removed and its unique subsets covered by adding one block
        improved = True
        while improved and time.time() < deadline - 4.0:
            improved = False
            # Try each selected block
            for i in range(len(selected_block_indices)):
                if time.time() > deadline - 4.0:
                    break
                b_idx = selected_block_indices[i]
                
                # Find subsets that would become uncovered if we remove this block
                unique_subsets = []
                for u in block_to_subsets[b_idx]:
                    if cover_count[u] == 1:
                        unique_subsets.append(u)
                
                if not unique_subsets:
                    # This block is redundant, remove it
                    selected_block_indices.pop(i)
                    for u in block_to_subsets[b_idx]:
                        cover_count[u] -= 1
                    improved = True
                    break
                
                # Try to find a single block that covers all unique_subsets
                # and is not already in the solution
                best_candidate = -1
                best_gain = 0
                for u in unique_subsets:
                    for b in subset_to_blocks[u]:
                        if b not in selected_block_indices:
                            # Check if this block covers all unique_subsets
                            covers_all = True
                            for u2 in unique_subsets:
                                if u2 not in block_to_subsets[b]:
                                    covers_all = False
                                    break
                            if covers_all:
                                # Count how many other subsets this block covers
                                gain = 0
                                for u2 in block_to_subsets[b]:
                                    if cover_count[u2] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_candidate = b
                
                if best_candidate >= 0:
                    # Remove the old block and add the new one
                    selected_block_indices.pop(i)
                    for u in block_to_subsets[b_idx]:
                        cover_count[u] -= 1
                    selected_block_indices.append(best_candidate)
                    for u in block_to_subsets[best_candidate]:
                        cover_count[u] += 1
                    improved = True
                    break
        
        # Rebuild blocks_found from selected_block_indices
        blocks_found = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    
    return blocks_found
