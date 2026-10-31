import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Map each block to the list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        indices = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(indices)

    # Reverse map: t_sub_idx -> list of block indices
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Convert to numpy arrays for vectorized operations where possible
    # block_to_tsubs is a list of lists. For v=19, k=7, t=4, each has 35 elements.
    
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    blocks_in_cover = []
    in_cover_set = set()

    def add_block(b_idx):
        for idx in block_to_tsubs[b_idx]:
            cover_count[idx] += 1
        blocks_in_cover.append(b_idx)
        in_cover_set.add(b_idx)

    def remove_block(b_idx):
        for idx in block_to_tsubs[b_idx]:
            cover_count[idx] -= 1
        if b_idx in in_cover_set:
            in_cover_set.remove(b_idx)
            try:
                blocks_in_cover.remove(b_idx)
            except ValueError:
                pass

    # 1. Greedy Initialization
    # Use a randomized greedy approach.
    # To speed up, we can maintain a list of uncovered t-subsets.
    # But checking gain for all 50k blocks is expensive.
    # Instead, pick a random uncovered t-subset and find the best block covering it.
    
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    
    # Optimization: Pre-compute a random permutation of blocks to break ties?
    # Or just use random.choice on candidates.
    
    while np.any(uncovered_mask):
        if time.time() > deadline - 10: break
        
        # Pick a random uncovered t-subset
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0: break
        target_idx = random.choice(uncovered_indices)
        
        # Candidates are blocks covering this target
        candidates = t_sub_to_blocks[target_idx]
        
        best_block_idx = -1
        max_gain = -1
        
        # Evaluate candidates
        # Gain = number of t-subsets in block with cover_count == 0
        # We can vectorize this slightly if we convert block_to_tsubs to arrays,
        # but for 455 candidates * 35 subsets, pure Python loop is ~16k ops, which is fast.
        for b_idx in candidates:
            gain = 0
            # Using numpy sum for speed?
            # gain = np.sum(cover_count[block_to_tsubs[b_idx]] == 0)
            # This creates a temp array. Let's try it.
            gain = np.sum(cover_count[block_to_tsubs[b_idx]] == 0)
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx != -1:
            add_block(best_block_idx)
            # Update uncovered mask
            # A t-subset becomes covered if its count goes from 0 to 1.
            # We can just recompute the mask or update it.
            # Recomputing is O(num_t_subsets). Updating is O(k).
            for t_idx in block_to_tsubs[best_block_idx]:
                if cover_count[t_idx] == 1:
                    uncovered_mask[t_idx] = False

    # 2. Local Search
    # Strategy: 
    # A. Delete redundant blocks.
    # B. Swap blocks to reduce "waste" (unique subsets).
    
    # Precompute sets for faster intersection? 
    # We have t_sub_to_blocks as lists.
    
    def get_unique_tsubs(b_idx):
        # Returns list of t_idx where cover_count is 1
        return [t_idx for t_idx in block_to_tsubs[b_idx] if cover_count[t_idx] == 1]

    def is_redundant(b_idx):
        for t_idx in block_to_tsubs[b_idx]:
            if cover_count[t_idx] == 1:
                return False
        return True

    # Local search loop
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 1:
        improved = False
        
        # Phase 1: Deletion
        # Iterate through a random order of blocks to remove redundant ones
        # We can't modify list while iterating, so make a copy or iterate indices
        # Since blocks_in_cover can be large, let's just iterate and check.
        # To be safe against removal during iteration, we iterate over a snapshot of indices?
        # Or just iterate over the list and skip if removed?
        # Better: iterate over list(blocks_in_cover)
        
        to_remove = []
        for b_idx in blocks_in_cover:
            if is_redundant(b_idx):
                to_remove.append(b_idx)
                
        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True
            
        if improved:
            no_improvement_count = 0
            continue # Restart as state changed significantly
            
        # Phase 2: Swaps
        # Find blocks with unique subsets (critical blocks)
        # We want to swap a critical block with another block that covers its unique subsets.
        
        # Identify critical blocks
        critical_blocks = []
        for b_idx in blocks_in_cover:
            if any(cover_count[t_idx] == 1 for t_idx in block_to_tsubs[b_idx]):
                critical_blocks.append(b_idx)
                
        if not critical_blocks:
            # If no block has unique subsets, all subsets are covered by >= 2 blocks.
            # Then any block is redundant? 
            # If all subsets of a block are covered by >= 2 blocks, then removing it leaves them at >= 1.
            # So it should have been removed in Phase 1.
            # If we are here, it means Phase 1 didn't find any? 
            # That implies is_redundant returned False for all?
            # is_redundant returns False if there is a subset with count 1.
            # So if critical_blocks is empty, no block has a subset with count 1.
            # So all subsets have count >= 2.
            # Then is_redundant should have returned True for all blocks.
            # So Phase 1 should have removed them.
            # Unless the list was empty?
            break
            
        # Pick a random critical block
        b_old_idx = random.choice(critical_blocks)
        unique_tsubs = get_unique_tsubs(b_old_idx)
        
        if not unique_tsubs:
            continue # Should not happen for critical block
            
        # Find a replacement block that covers all unique_tsubs
        # Intersection of t_sub_to_blocks[t_idx] for all t_idx in unique_tsubs
        
        # Start with the smallest candidate list for efficiency?
        # Or just iterate.
        
        # Get candidate blocks
        # Since unique_tsubs can be up to 35, intersection might be empty.
        
        # Optimization: Sort unique_tsubs by length of t_sub_to_blocks to prune early?
        # t_sub_to_blocks lengths are roughly constant (455).
        
        candidate_blocks = set(t_sub_to_blocks[unique_tsubs[0]])
        
        for t_idx in unique_tsubs[1:]:
            # Intersect
            # Using set intersection
            candidate_blocks.intersection_update(t_sub_to_blocks[t_idx])
            if not candidate_blocks:
                break
        
        if candidate_blocks:
            # We want a block that is NOT b_old_idx
            candidate_blocks.discard(b_old_idx)
            
            if candidate_blocks:
                # Prefer blocks not currently in the cover?
                # Swapping with a block already in the cover is equivalent to removal if the new block covers the unique parts.
                # But if b_new is in cover, and we remove b_old, we just deleted b_old.
                # This is valid if b_new covers all unique parts of b_old.
                
                # Let's pick a random candidate
                b_new_idx = random.choice(list(candidate_blocks))
                
                # Perform swap
                remove_block(b_old_idx)
                add_block(b_new_idx)
                
                improved = True
                no_improvement_count = 0
                continue # Restart loop
                
        # If no swap found for this block, try another?
        # For now, just count as no improvement for this iteration
        no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Try a more aggressive restart or break?
            # Let's try a random restart of the local search order or break if stuck too long.
            # To avoid infinite loop, we can break if we've tried many times.
            # Or we can try to force a swap by relaxing the "covers ALL unique" constraint?
            # No, that breaks coverage.
            
            # Let's try to find a swap for ANY critical block in a batch?
            # For now, just break to avoid wasting time.
            # Actually, maybe we should try other critical blocks.
            # Let's pick a different random critical block next time.
            # The loop will pick a new random one.
            # If we keep failing, we will loop until deadline.
            # That's acceptable as long as we don't do expensive work.
            # But checking intersection is cheap.
            
            # To avoid getting stuck in a local minimum where no single swap works,
            # we might need 2-swaps. But that's complex.
            
            # Let's just continue.
            pass

    # Final check: Ensure coverage is valid?
    # The greedy + local search should maintain validity.
    # But just in case, we can verify.
    # Verification is O(num_t_subsets).
    
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in blocks_in_cover]
    
    # Sort blocks for deterministic output
    final_blocks.sort()
    
    return final_blocks
