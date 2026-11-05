"""Vectorized greedy set-cover + exact pruning for covering design C(24,7,3)."""
import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (blocks) as a matrix of shape (n_blocks, k)
    # For v=24, k=7: C(24,7) = 346104 blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    blocks_arr = np.array(all_blocks, dtype=np.int32)  # shape (n_blocks, k)

    # Precompute for each block which t-subsets it covers
    # Each block covers C(k,t) t-subsets. For k=7,t=3: 35 per block.
    # We'll build a coverage matrix: cov[block_idx, tsub_idx] = 1 if block covers tsub
    # This is a huge matrix (346104 x 2024) but we can use sparse-like approach.
    # Instead, for each t-subset, precompute which blocks cover it.
    # Actually, let's build: for each block, list of t-subset indices it covers.
    
    # Better approach: for greedy, we need to quickly compute gain of a block.
    # gain(block) = number of uncovered t-subsets in block's coverage.
    # We'll maintain a boolean array of covered t-subsets.
    
    # Precompute block_to_tsubs: for each block, the list of t-subset indices
    block_to_tsubs = []
    for b in all_blocks:
        tsubs = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(tsubs)
    
    # For faster access, convert to numpy array (ragged -> use list of arrays)
    # Actually for v=24,k=7,t=3, each block has exactly 35 t-subsets.
    # So we can make a 2D array: block_tsubs[i] = indices of 35 t-subsets for block i
    block_tsubs_arr = np.array(block_to_tsubs, dtype=np.int32)  # shape (n_blocks, 35)
    
    covered = np.zeros(n_tsubs, dtype=bool)
    selected_blocks = []
    
    # Greedy set cover: repeatedly pick block with max uncovered coverage
    while not covered.all():
        if time.time() > deadline - 10.0:
            break
        
        # For each block, count how many of its t-subsets are uncovered
        # covered[block_tsubs_arr[i]] gives boolean array of which are covered
        # We want count of False (uncovered)
        # This is: n_per_block - sum(covered[block_tsubs_arr[i]])
        
        # Vectorized: for all blocks at once
        # block_tsubs_arr shape (n_blocks, 35), covered shape (n_tsubs,)
        # covered[block_tsubs_arr] shape (n_blocks, 35)
        is_covered = covered[block_tsubs_arr]  # shape (n_blocks, 35)
        gains = (~is_covered).sum(axis=1)  # shape (n_blocks,)
        
        # Pick block with max gain
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break  # No improvement possible
        
        selected_blocks.append(all_blocks[best_idx])
        
        # Mark its t-subsets as covered
        covered[block_tsubs_arr[best_idx]] = True
    
    # If we didn't achieve full coverage (due to time), try to finish with random blocks
    if not covered.all():
        uncovered = np.where(~covered)[0]
        # For each uncovered t-subset, find a block that covers it
        # Build reverse map: t-subset -> list of blocks covering it
        # This is expensive to build fully, so do it on-demand
        tsub_to_blocks = {}
        for i, tsub in enumerate(all_tsubs):
            if not covered[i]:
                # Find blocks containing this t-subset
                tsub_set = set(tsub)
                candidates = []
                for j, b in enumerate(all_blocks):
                    if tsub_set.issubset(set(b)):
                        candidates.append(j)
                tsub_to_blocks[i] = candidates
        
        for i in uncovered:
            if time.time() > deadline - 5.0:
                break
            if i in tsub_to_blocks and tsub_to_blocks[i]:
                # Pick a block that maximizes additional coverage
                best_j = tsub_to_blocks[i][0]
                best_gain = 0
                for j in tsub_to_blocks[i][:50]:  # Limit search
                    g = (~covered[block_tsubs_arr[j]]).sum()
                    if g > best_gain:
                        best_gain = g
                        best_j = j
                selected_blocks.append(all_blocks[best_j])
                covered[block_tsubs_arr[best_j]] = True
    
    # Prune redundant blocks
    # A block is redundant if removing it doesn't uncover any t-subset
    # i.e., all t-subsets in the block are covered by other selected blocks
    n_selected = len(selected_blocks)
    if n_selected > 0:
        # Build coverage count: for each t-subset, how many selected blocks cover it
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in selected_blocks:
            tsub_indices = block_to_tsubs[all_blocks.index(b)]
            cov_count[tsub_indices] += 1
        
        # Iteratively remove blocks where all their t-subsets have cov_count >= 2
        changed = True
        while changed:
            changed = False
            new_selected = []
            for i, b in enumerate(selected_blocks):
                if time.time() > deadline - 3.0:
                    new_selected.append(b)
                    continue
                tsub_indices = block_to_tsubs[all_blocks.index(b)]
                # Check if all t-subsets in this block are covered by other blocks
                # i.e., cov_count[tsub_indices] >= 2 for all
                if np.all(cov_count[tsub_indices] >= 2):
                    # This block is redundant, remove it
                    cov_count[tsub_indices] -= 1
                    changed = True
                else:
                    new_selected.append(b)
            selected_blocks = new_selected
    
    # Verify coverage
    final_covered = np.zeros(n_tsubs, dtype=bool)
    for b in selected_blocks:
        tsub_indices = block_to_tsubs[all_blocks.index(b)]
        final_covered[tsub_indices] = True
    
    # If not fully covered (shouldn't happen), add blocks to fix
    if not final_covered.all():
        uncovered = np.where(~final_covered)[0]
        for i in uncovered:
            if time.time() > deadline - 1.0:
                break
            tsub = all_tsubs[i]
            tsub_set = set(tsub)
            # Find any block containing this t-subset
            for j, b in enumerate(all_blocks):
                if tsub_set.issubset(set(b)):
                    if b not in selected_blocks:
                        selected_blocks.append(b)
                    break
            # Update coverage
            if selected_blocks:
                last_b = selected_blocks[-1]
                tsub_indices = block_to_tsubs[all_blocks.index(last_b)]
                final_covered[tsub_indices] = True
    
    return [list(b) for b in selected_blocks]
