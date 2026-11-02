import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the indices of t-subsets it contains
    block_t_indices = []
    # Also store for each t-subset, the list of blocks containing it (for local search)
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        t_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            t_indices.append(idx)
            t_subset_to_blocks[idx].append(b_idx)
        block_t_indices.append(t_indices)

    def time_check():
        return time.time() > deadline

    # 2. Greedy Initialization
    # We want to cover all t-subsets.
    # Objective: Maximize number of newly covered t-subsets.
    counts = [0] * num_t_subsets
    blocks = []
    
    # Optimization: Keep a list of "active" blocks? No, just iterate all.
    # To speed up, we can break early if we find a block covering all remaining?
    # Max possible coverage for a block is C(k,t) = 56.
    # If remaining_uncovered <= 56, we might find a perfect fit, but unlikely to help much globally.
    
    remaining_uncovered = num_t_subsets
    
    while remaining_uncovered > 0:
        if time_check():
            break
            
        best_block = -1
        max_new_cov = -1
        
        # Iterate through all blocks
        for b_idx in range(num_blocks):
            new_cov = 0
            # Check how many t-subsets in this block are currently uncovered
            for idx in block_t_indices[b_idx]:
                if counts[idx] == 0:
                    new_cov += 1
                    # Early termination if we can't beat current best?
                    # If new_cov + (len(block_t_indices[b_idx]) - current_idx) <= max_new_cov, break?
                    # Hard to do easily without loop index.
            
            if new_cov > max_new_cov:
                max_new_cov = new_cov
                best_block = b_idx
                # If we found a block covering all remaining? 
                # If max_new_cov == remaining_uncovered, we can't do better.
                if max_new_cov == remaining_uncovered:
                    break
        
        if best_block == -1 or max_new_cov == 0:
            # Should not happen unless remaining_uncovered is 0
            break
            
        # Add the best block
        blocks.append(best_block)
        for idx in block_t_indices[best_block]:
            if counts[idx] == 0:
                remaining_uncovered -= 1
            counts[idx] += 1

    # 3. Local Search (Pruning and Swapping)
    # Goal: Minimize len(blocks)
    
    def local_search():
        nonlocal blocks, counts
        improved = True
        while improved and not time_check():
            improved = False
            
            # Pass 1: Remove redundant blocks
            # A block is redundant if all its t-subsets are covered by other blocks.
            # i.e., for all idx in block_t_indices[b], counts[idx] > 1.
            
            new_blocks = []
            for b_idx in blocks:
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            
            blocks = new_blocks
            
            if time_check():
                break
                
            # Pass 2: Kick/Swap
            # Try to swap a block with another to reduce total count? 
            # Or just try to remove a block and re-cover the lost subsets with one other block.
            # This is a 1-for-1 swap.
            
            # Shuffle blocks to try different ones
            random.shuffle(blocks)
            
            for b_out_idx in range(len(blocks)):
                if time_check():
                    break
                    
                b_out = blocks[b_out_idx]
                # Find t-subsets unique to b_out
                lost_indices = [idx for idx in block_t_indices[b_out] if counts[idx] == 1]
                
                if not lost_indices:
                    # It was redundant, should have been caught in Pass 1?
                    # Maybe counts changed? No.
                    # But if Pass 1 didn't run or failed? 
                    # If lost_indices is empty, we can just remove it.
                    for idx in block_t_indices[b_out]:
                        counts[idx] -= 1
                    blocks.pop(b_out_idx)
                    improved = True
                    break # Restart loop as blocks changed
                    
                # Find a block b_in that covers all lost_indices
                # Intersection of t_subset_to_blocks for all idx in lost_indices
                candidate_blocks = None
                for idx in lost_indices:
                    if candidate_blocks is None:
                        candidate_blocks = set(t_subset_to_blocks[idx])
                    else:
                        candidate_blocks.intersection_update(t_subset_to_blocks[idx])
                
                if not candidate_blocks:
                    continue
                
                # Exclude b_out itself (if it was in candidates, which it is)
                candidate_blocks.discard(b_out)
                
                if not candidate_blocks:
                    continue
                
                # Pick a random candidate
                b_in = random.choice(list(candidate_blocks))
                
                # Check if b_in is already in blocks? 
                # If it is, we are just moving it? No, we remove b_out and add b_in.
                # If b_in is already in blocks, we just remove b_out. 
                # But b_in covers lost_indices, so counts will stay >= 1.
                # This effectively removes b_out.
                
                # Perform swap
                # Remove b_out
                for idx in block_t_indices[b_out]:
                    counts[idx] -= 1
                blocks.pop(b_out_idx)
                
                # Add b_in (if not already present? 
                # If b_in is in blocks, we don't add it again, we just keep the existing one.
                # But we need to increment counts? 
                # Wait, if b_in is already in blocks, its counts are already accounted for.
                # If we just remove b_out, the lost_indices are still covered by b_in.
                # So we don't need to add b_in again.
                
                if b_in not in blocks:
                    for idx in block_t_indices[b_in]:
                        counts[idx] += 1
                    blocks.append(b_in)
                
                improved = True
                break # Restart loop as blocks changed
                
        return blocks

    local_search()
    
    # Convert indices back to lists
    result = [list(all_blocks[b_idx]) for b_idx in blocks]
    
    return result
