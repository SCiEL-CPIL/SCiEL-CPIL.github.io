import itertools
import time
import numpy as np

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
    # block_subsets as a list of arrays
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]
    
    # Calculate weights: 1 / number of blocks covering the subset
    # This is the Gonzalez-Park heuristic weight
    weights = np.zeros(n_subsets, dtype=np.float64)
    for u in range(n_subsets):
        weights[u] = 1.0 / len(subset_to_blocks[u])

    # Initialize gain vector: gain[b] = sum of weights of uncovered subsets in block b
    uncovered = np.ones(n_subsets, dtype=bool)
    gains = np.zeros(n_blocks, dtype=np.float64)

    # Initial gains: for each subset, add weight to all blocks containing it
    for u in range(n_subsets):
        gains[subset_blocks_arr[u]] += weights[u]

    # Coverage count for removal pass
    coverage_count = np.zeros(n_subsets, dtype=np.int32)

    blocks_found = []
    used_blocks = set()

    while True:
        if time.time() > deadline - 5.0:
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
        
        # Update coverage count
        for u in block_to_subsets[b_star]:
            coverage_count[u] += 1

        # Update uncovered and gains
        # Identify newly covered subsets
        newly_covered = []
        for u in block_to_subsets[b_star]:
            if uncovered[u]:
                newly_covered.append(u)
        
        if newly_covered:
            uncovered[newly_covered] = False
            
            # Vectorized gain update
            for u in newly_covered:
                gains[subset_blocks_arr[u]] -= weights[u]

        # Early exit if all covered
        if not uncovered.any():
            break

    # Polish pass: Remove redundant blocks
    # We can try to remove blocks that do not cover any unique subset
    # i.e., all subsets in the block are covered by at least one other block
    
    # We iterate through the blocks found and try to remove them
    # If we remove a block, we must decrement coverage_count
    # We do this in a single pass for safety, but could do multiple passes if time permits
    
    # Sort blocks by index? Or just iterate?
    # Let's iterate through the list of block indices
    # Note: blocks_found contains the block contents (lists of points), we need indices
    # We stored used_blocks as indices.
    
    # Reconstruct list of indices for removal logic
    # Actually, we can just iterate through used_blocks
    # But we need to be careful: removing a block might make another block removable.
    # A single pass is conservative.
    
    removal_candidates = list(used_blocks)
    
    for b_idx in removal_candidates:
        if time.time() > deadline - 2.0:
            break
            
        # Check if block b_idx is redundant
        # i.e., for all u in block_to_subsets[b_idx], coverage_count[u] > 1
        is_redundant = True
        for u in block_to_subsets[b_idx]:
            if coverage_count[u] <= 1:
                is_redundant = False
                break
        
        if is_redundant:
            # Remove block
            # Decrement coverage count
            for u in block_to_subsets[b_idx]:
                coverage_count[u] -= 1
            
            # Remove from blocks_found
            # Find the block content
            block_content = list(all_blocks[b_idx])
            if block_content in blocks_found:
                blocks_found.remove(block_content)
            used_blocks.remove(b_idx)

    return blocks_found
