"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precomputation: all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    
    # Precompute: for each point, which t-subsets contain it
    tsubsets_by_point = [[] for _ in range(v)]
    for i, s in enumerate(all_tsubsets):
        for x in s:
            tsubsets_by_point[x].append(i)
    
    # Precompute: for each 7-block, which t-subsets it covers (as a bitmask)
    # Using bitsets for fast coverage tracking
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-subset indices it covers
    block_tsubsets = []
    for block in all_blocks:
        idxs = []
        for ts in itertools.combinations(block, t):
            idxs.append(tsubset_to_idx[ts])
        block_tsubsets.append(np.array(idxs, dtype=np.int32))
    
    # Coverage tracking: boolean array
    covered = np.zeros(n_tsubsets, dtype=np.uint8)
    
    blocks = []
    block_indices = []  # indices into all_blocks
    
    # Phase 1: Greedy set cover with max-gain selection
    # We'll iteratively pick the block that covers the most uncovered t-subsets
    while True:
        if time.time() > deadline - 30.0:  # Reserve 30s for local search
            break
            
        uncovered_count = n_tsubsets - np.sum(covered)
        if uncovered_count == 0:
            break
        
        # For each candidate block, compute how many currently-uncovered t-subsets it covers
        # Vectorized approach: for each block, count uncovered t-subsets in its coverage set
        best_block_idx = -1
        best_gain = 0
        
        # Process blocks in batches for speed
        batch_size = 2000
        for batch_start in range(0, n_blocks, batch_size):
            if time.time() > deadline - 30.0:
                break
            batch_end = min(batch_start + batch_size, n_blocks)
            
            batch_gains = np.zeros(batch_end - batch_start, dtype=np.int32)
            for j in range(batch_end - batch_start):
                bi = batch_start + j
                ts_idxs = block_tsubsets[bi]
                # Count how many of these t-subsets are currently uncovered
                batch_gains[j] = np.sum(1 - covered[ts_idxs])
            
            local_max_idx = np.argmax(batch_gains)
            local_max_gain = batch_gains[local_max_idx]
            
            if local_max_gain > best_gain:
                best_gain = local_max_gain
                best_block_idx = batch_start + local_max_idx
        
        if best_block_idx == -1 or best_gain == 0:
            # Fallback: pick any block that covers at least one uncovered t-subset
            # This shouldn't happen if coverage is incomplete
            for bi in range(n_blocks):
                ts_idxs = block_tsubsets[bi]
                if np.any(1 - covered[ts_idxs]):
                    best_block_idx = bi
                    best_gain = 1
                    break
            if best_block_idx == -1:
                break
        
        # Select this block
        blocks.append(list(all_blocks[best_block_idx]))
        block_indices.append(best_block_idx)
        
        # Mark covered
        ts_idxs = block_tsubsets[best_block_idx]
        covered[ts_idxs] = 1
    
    if time.time() > deadline - 5.0:
        return blocks
    
    # Phase 2: Local search - try to reduce block count
    # Strategy: for each block in our solution, try removing it and see if we can
    # re-cover the lost t-subsets with fewer new blocks
    # Simpler approach: try swapping one block for another that covers more
    # Or: try removing a block and greedily re-covering the gap
    
    n_solution = len(blocks)
    
    # Try to remove blocks one by one and re-cover
    # Sort blocks by "importance" (fewest unique t-subsets they cover)
    # A block is "removable" if all its t-subsets are covered by other blocks
    
    def can_remove_block(block_idx_in_solution):
        """Check if removing this block still leaves all t-subsets covered."""
        # Compute coverage without this block
        # This is expensive, so we'll use a simpler heuristic:
        # Count how many t-subsets are ONLY covered by this block
        block = blocks[block_idx_in_solution]
        ts_idxs = block_tsubsets[block_indices[block_idx_in_solution]]
        
        # For each t-subset in this block, check if it's covered by another block
        for ts_idx in ts_idxs:
            ts = all_tsubsets[ts_idx]
            # Check if any other block in our solution covers this t-subset
            found = False
            for other_idx in range(n_solution):
                if other_idx == block_idx_in_solution:
                    continue
                other_block = blocks[other_idx]
                if all(x in other_block for x in ts):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Try removing blocks iteratively
    removed_any = True
    while removed_any and time.time() < deadline - 2.0:
        removed_any = False
        for i in range(len(blocks)):
            if time.time() > deadline - 2.0:
                break
            if can_remove_block(i):
                blocks.pop(i)
                block_indices.pop(i)
                n_solution = len(blocks)
                removed_any = True
                break  # Restart from beginning since indices changed
    
    # Try swapping: replace a block with another that covers more uncovered t-subsets
    # after removal
    if time.time() < deadline - 2.0:
        # Recompute coverage
        covered = np.zeros(n_tsubsets, dtype=np.uint8)
        for bi in block_indices:
            ts_idxs = block_tsubsets[bi]
            covered[ts_idxs] = 1
        
        # For each block, see if we can replace it with a better one
        # that covers all its t-subsets plus more
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    break
                current_block = blocks[i]
                current_ts_idxs = set(block_tsubsets[block_indices[i]])
                
                # Find best replacement block that covers all current t-subsets
                # and has maximum total coverage
                best_replacement = -1
                best_total_cover = 0
                
                for bi in range(n_blocks):
                    if bi == block_indices[i]:
                        continue
                    replacement_ts_idxs = set(block_tsubsets[bi])
                    # Check if replacement covers all current t-subsets
                    if not current_ts_idxs.issubset(replacement_ts_idxs):
                        continue
                    # Count total coverage (unique t-subsets covered by this block)
                    total = len(replacement_ts_idxs)
                    if total > best_total_cover:
                        best_total_cover = total
                        best_replacement = bi
                
                if best_replacement != -1 and best_replacement != block_indices[i]:
                    blocks[i] = list(all_blocks[best_replacement])
                    block_indices[i] = best_replacement
                    improved = True
    
    # Final coverage check and repair if needed
    covered = np.zeros(n_tsubsets, dtype=np.uint8)
    for bi in block_indices:
        ts_idxs = block_tsubsets[bi]
        covered[ts_idxs] = 1
    
    # If not fully covered, add more blocks greedily
    while True:
        if time.time() > deadline - 0.5:
            break
        uncovered_mask = 1 - covered
        if np.sum(uncovered_mask) == 0:
            break
        
        best_block_idx = -1
        best_gain = 0
        for bi in range(n_blocks):
            ts_idxs = block_tsubsets[bi]
            gain = np.sum(uncovered_mask[ts_idxs])
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
        
        if best_block_idx == -1 or best_gain == 0:
            break
        
        blocks.append(list(all_blocks[best_block_idx]))
        block_indices.append(best_block_idx)
        ts_idxs = block_tsubsets[best_block_idx]
        covered[ts_idxs] = 1
    
    return blocks
# EVOLVE-BLOCK-END
