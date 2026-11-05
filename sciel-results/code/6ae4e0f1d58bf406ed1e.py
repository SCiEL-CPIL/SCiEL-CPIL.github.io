"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    # Leave a safety margin for the final validity check and return
    deadline = start_time + 110.0 

    # 1. Precompute all t-subsets and map to indices
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Build incidence structures
    # block_to_subsets[b] = list of subset indices covered by block b
    block_to_subsets = [[] for _ in range(num_blocks)]
    # subset_to_blocks[u] = list of block indices covering subset u
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    # This precomputation takes a few seconds for v=24, k=7, t=3
    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access if needed, but lists are fine for iteration
    # We keep them as lists for simplicity, but use numpy for the gain vector
    
    # 4. Initialize state
    # uncovered[u] = True if subset u is not yet covered
    uncovered = np.ones(num_subsets, dtype=bool)
    # gains[b] = number of currently uncovered subsets covered by block b
    gains = np.zeros(num_blocks, dtype=np.int32)
    
    # Initial gains: for each subset, add 1 to all blocks containing it
    # This is O(num_subsets * avg_blocks_per_subset)
    for u in range(num_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1
            
    blocks_found = []
    used_blocks = set()
    
    # 5. Greedy Set Cover
    while True:
        if time.time() > deadline - 10.0:  # Reserve 10s for cleanup
            break
            
        if not uncovered.any():
            break
            
        # Find block with max gain
        b_star = int(np.argmax(gains))
        
        if gains[b_star] <= 0:
            # No block can cover any remaining uncovered subset?
            # This shouldn't happen in a valid covering design problem unless we ran out of blocks
            # or there's a bug. But if it happens, break to avoid infinite loop.
            break
            
        # Add this block to solution
        blocks_found.append(list(all_blocks[b_star]))
        used_blocks.add(b_star)
        
        # Update uncovered and gains
        # Get the subsets covered by this block
        covered_subsets = block_to_subsets[b_star]
        
        for u in covered_subsets:
            if uncovered[u]:
                uncovered[u] = False
                # Decrement gain for all blocks that also cover this subset u
                for b in subset_to_blocks[u]:
                    gains[b] -= 1
                    
        # Early exit if all covered
        if not uncovered.any():
            break

    # 6. Redundancy Removal
    # Try to remove blocks that are not essential.
    # A block is redundant if every t-subset it contains is covered by at least one other block in the solution.
    # We do this in reverse order (last added first) as they are likely to be less "core".
    
    # To speed this up, we can maintain a count of how many times each subset is covered.
    # However, rebuilding this count for every removal is O(N_blocks * C(k,t)).
    # C(7,3) = 35. N_blocks ~ 76. 76 * 35 = 2660 operations per check.
    # Checking all blocks: 76 * 2660 ~ 200k operations. This is fast.
    
    if time.time() < deadline - 5.0:
        # We will iterate multiple passes to catch chains of redundancies
        max_passes = 10
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
                
            removed_any = False
            # Iterate backwards
            for i in range(len(blocks_found) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                    
                block_to_remove = blocks_found[i]
                # Check if this block is redundant
                is_redundant = True
                
                # For each t-subset in this block, check if it's covered by another block
                for ts in itertools.combinations(block_to_remove, t):
                    ts_idx = subset_to_idx[ts]
                    # Count how many blocks in blocks_found (excluding i) cover ts_idx
                    # Optimization: precompute cover counts? 
                    # Let's just iterate.
                    covered_by_other = False
                    for j, other_block in enumerate(blocks_found):
                        if j == i:
                            continue
                        # Check if other_block contains ts
                        # Since blocks are sorted lists, we can use a set or simple check
                        # But for small k=7, a simple check is fast.
                        # Even better: use the precomputed block_to_subsets
                        # If ts_idx is in block_to_subsets[j], then it's covered.
                        # But checking membership in a list is O(N).
                        # Let's use a set of covered subsets for the current solution?
                        # That's expensive to maintain.
                        
                        # Alternative: Just check if any other block contains the 3 points.
                        # This is O(1) per block if we use sets, but we have lists.
                        # Let's stick to the simple check for now, it's fast enough for ~76 blocks.
                        if set(ts).issubset(set(other_block)):
                            covered_by_other = True
                            break
                            
                    if not covered_by_other:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    blocks_found.pop(i)
                    removed_any = True
                    
            if not removed_any:
                break

    # 7. Final Validity Check and Repair
    # Ensure all t-subsets are covered.
    if time.time() < deadline - 1.0:
        # Build a set of all covered t-subsets
        covered_set = set()
        for b in blocks_found:
            for ts in itertools.combinations(b, t):
                covered_set.add(ts)
                
        # Find missing subsets
        all_subsets_set = set(all_subsets)
        missing = all_subsets_set - covered_set
        
        # If there are missing subsets, add blocks to cover them
        # This should ideally not happen if the greedy phase was complete,
        # but redundancy removal might have broken coverage if logic was flawed (it shouldn't).
        # Also, if the greedy phase timed out, we might have missing subsets.
        
        if missing:
            # Convert missing back to indices for gain calculation if we were to continue greedy
            # But simpler: just greedily add blocks for missing subsets
            missing_list = list(missing)
            # Sort missing to have deterministic behavior
            missing_list.sort()
            
            # We need to re-calculate gains based on the current set of blocks?
            # Or just pick a block that covers the most missing subsets.
            
            # Let's just iterate through missing subsets and find a block that covers it
            # and has the most other missing subsets.
            
            # For speed, just pick the first missing subset and find a block containing it
            # that covers the most other missing subsets.
            
            # Re-initialize a local gain for missing subsets only?
            # Let's just do a simple greedy repair.
            
            while missing and time.time() < deadline - 0.5:
                first_missing = missing_list[0] # Or pick random
                # Find best block covering first_missing
                best_block_idx = -1
                best_count = -1
                
                # Which blocks cover first_missing?
                fm_idx = subset_to_idx[first_missing]
                candidate_blocks = subset_to_blocks[fm_idx]
                
                for b_idx in candidate_blocks:
                    # Count how many missing subsets this block covers
                    count = 0
                    for u_idx in block_to_subsets[b_idx]:
                        # Check if u_idx corresponds to a missing subset
                        # We need a mapping from idx to subset or a set of missing idxs
                        pass # This is getting complex. Let's use a simpler repair.
                
                # Simpler repair: Just add a block that contains the first missing subset.
                # To be efficient, pick the one that covers the most *total* uncovered (missing) subsets.
                
                # Let's create a set of missing indices for fast lookup
                missing_indices = {subset_to_idx[m] for m in missing}
                
                best_b = -1
                max_gain = -1
                for b_idx in candidate_blocks:
                    gain = 0
                    for u_idx in block_to_subsets[b_idx]:
                        if u_idx in missing_indices:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_b = b_idx
                        
                if best_b != -1:
                    new_block = list(all_blocks[best_b])
                    # Avoid duplicates
                    if new_block not in blocks_found:
                        blocks_found.append(new_block)
                        # Update missing
                        for u_idx in block_to_subsets[best_b]:
                            if u_idx in missing_indices:
                                missing.discard(all_subsets[u_idx])
                        missing_list = list(missing)
                    else:
                        # If block already exists, it means our coverage check was wrong?
                        # Or we are stuck. Break to avoid infinite loop.
                        break
                else:
                    break

    # 8. Final Deduplication
    seen = set()
    unique_blocks = []
    for b in blocks_found:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
            
    return unique_blocks

# EVOLVE-BLOCK-END
