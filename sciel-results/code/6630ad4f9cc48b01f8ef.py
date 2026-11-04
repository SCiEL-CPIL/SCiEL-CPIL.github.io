import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Generate all k-subsets (blocks)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # For each k-subset, find the indices of the t-subsets it covers
    # block_covers[i] will be a list of t-subset indices covered by block i
    block_covers = []
    for block in all_k_subs:
        covered_indices = []
        for sub in itertools.combinations(block, t):
            covered_indices.append(t_sub_to_idx[sub])
        block_covers.append(covered_indices)
    
    # Convert to NumPy array for vectorization
    # All lists have length C(k, t)
    t_in_block = len(block_covers[0])
    block_covers_np = np.array(block_covers, dtype=np.int32) # shape (n_k, t_in_block)
    
    # Precompute for each t-subset, which blocks cover it
    # t_sub_to_blocks[i] is a list of block indices that cover t-subset i
    t_sub_to_blocks = defaultdict(list)
    for b_idx, covered in enumerate(block_covers):
        for t_idx in covered:
            t_sub_to_blocks[t_idx].append(b_idx)
    
    # Helper to convert block index to sorted list of points
    # all_k_subs is already sorted tuples
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks_used = []
        
        while uncovered.any():
            if time.time() > deadline - 5:
                break
            
            # Calculate gains for all blocks
            # uncovered[block_covers_np] gives a boolean matrix (n_k, t_in_block)
            # Sum along axis 1 gives count of newly covered subsets
            gains = uncovered[block_covers_np].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if uncovered.any() is true
                break
            
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = random.choice(candidates)
            
            # Update uncovered
            uncovered[block_covers_np[chosen_idx]] = False
            
            blocks_used.append(all_k_subs[chosen_idx])
            
        return blocks_used

    def reduce_cover(blocks):
        if not blocks:
            return blocks
        
        # Map block tuple to its index in 'blocks' list for quick lookup?
        # Or just compute coverage counts.
        # coverage_count[i] = number of blocks in 'blocks' that cover t-subset i
        coverage_count = np.zeros(n_t, dtype=np.int32)
        
        # Precompute set of blocks for O(1) lookup?
        # We need to check if a block is in the current cover.
        # Since we are modifying the list, it's better to use a set of tuples.
        blocks_set = set(blocks)
        
        # Calculate initial coverage
        for block in blocks:
            # Find index of this block in all_k_subs?
            # We don't have a direct map from tuple to index in all_k_subs.
            # We can use t_sub_to_blocks? No.
            # We can just iterate through the t-subsets of this block.
            for sub in itertools.combinations(block, t):
                coverage_count[t_sub_to_idx[sub]] += 1
        
        new_blocks = []
        for block in blocks:
            # Check if block is redundant
            # A block is redundant if every t-subset it covers is covered by at least one OTHER block.
            # i.e., coverage_count[sub] > 1 for all sub in block.
            is_redundant = True
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if coverage_count[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                # Update coverage counts
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    coverage_count[idx] -= 1
            else:
                new_blocks.append(block)
                
        return new_blocks

    def local_search(blocks):
        # Try to replace blocks to allow further reduction
        # Iterate through blocks
        # For each block, find t-subsets uniquely covered by it.
        # If any, try to find a replacement block that covers all unique subsets.
        
        # First, compute coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                coverage_count[t_sub_to_idx[sub]] += 1
                
        improved = True
        while improved:
            improved = False
            # Recompute coverage if we made changes? 
            # For simplicity, we do one pass of swaps, then reduce.
            
            for i, block in enumerate(blocks):
                if time.time() > deadline - 5:
                    return blocks
                
                # Find unique subsets for this block
                unique_subs = []
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    if coverage_count[idx] == 1:
                        unique_subs.append(idx)
                
                if not unique_subs:
                    continue
                
                # Try to find a replacement
                # The replacement must cover all unique_subs.
                # Candidates: Intersection of t_sub_to_blocks[unique_sub] for all unique_subs.
                # Start with the first unique sub's blocks
                candidates = set(t_sub_to_blocks[unique_subs[0]])
                
                for us in unique_subs[1:]:
                    candidates.intersection_update(t_sub_to_blocks[us])
                    
                if not candidates:
                    continue
                    
                # Remove current block from candidates (it's already there, but we want a *different* block)
                # Actually, the current block IS in candidates. We want a block NOT in the current cover?
                # Or just a different block?
                # If we swap B_i with B_j, B_j must not be in the cover (otherwise we have duplicates).
                blocks_set = set(blocks)
                
                found_replacement = False
                for cand_idx in candidates:
                    cand_block = all_k_subs[cand_idx]
                    if cand_block in blocks_set:
                        continue
                    
                    # Check if cand_block covers all unique_subs
                    # It does by construction of candidates intersection.
                    
                    # Perform swap
                    # 1. Remove B_i
                    for sub in itertools.combinations(block, t):
                        coverage_count[t_sub_to_idx[sub]] -= 1
                        
                    # 2. Add B_j
                    for sub in itertools.combinations(cand_block, t):
                        coverage_count[t_sub_to_idx[sub]] += 1
                        
                    blocks[i] = cand_block
                    blocks_set.discard(block)
                    blocks_set.add(cand_block)
                    
                    found_replacement = True
                    improved = True
                    break # Restart loop? Or just continue? 
                    # Restarting is safer to re-evaluate other blocks with new counts.
                    # But restarting the while loop is expensive.
                    # Let's just break the for loop and restart the while loop.
                    break
                
                if found_replacement:
                    break # Break for loop to restart while loop
                    
        # After swaps, run reduction
        return reduce_cover(blocks)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy instances
    for seed in range(100):
        if time.time() > deadline - 10:
            break
            
        blocks = run_greedy(seed)
        
        if not blocks:
            continue
            
        # Reduce
        blocks = reduce_cover(blocks)
        
        # Local Search
        if time.time() < deadline - 5:
            blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit a very good score, we can stop early?
        # LB is 264. If we get close to 300, keep going.
        
    if best_blocks is None:
        # Fallback: simple greedy without local search
        blocks = run_greedy(0)
        best_blocks = reduce_cover(blocks)
        
    return best_blocks
