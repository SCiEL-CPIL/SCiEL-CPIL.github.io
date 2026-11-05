import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using GRASP (Greedy Randomized Adaptive Search Procedure).
    """
    start_time = time.time()
    time_limit = 110  # Leave some buffer for Python overhead
    
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-subset indices for each block
    # C(k, t) is the number of t-subsets in a k-block
    num_t_in_block = len(list(itertools.combinations(range(k), t)))
    block_t_indices = np.zeros((num_blocks, num_t_in_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        # Get t-subsets of this block
        # Using combinations on the block values
        block_t_vals = list(itertools.combinations(block, t))
        for j, ts in enumerate(block_t_vals):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Helper to get indices for a specific block tuple quickly
    # We can use the precomputed array if we map block->index
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    best_blocks = None
    best_count = float('inf')
    
    def grasp_iteration():
        """
        Performs one GRASP iteration: Randomized Greedy Construction + Local Search (Deletion).
        Returns the list of blocks.
        """
        # Initialize state
        uncovered = np.ones(num_t_subs, dtype=np.int8)
        used_block_mask = np.zeros(num_blocks, dtype=bool)
        blocks_used_indices = []
        
        # Randomized Greedy Construction
        while uncovered.sum() > 0:
            # Calculate gains for all unused blocks
            # gains[i] = number of uncovered t-subsets covered by block i
            gains = uncovered[block_t_indices].sum(axis=1)
            
            # Mask used blocks
            gains[used_block_mask] = -1
            
            # Get max gain
            max_gain = np.max(gains)
            if max_gain <= 0:
                # Should not happen if covering is possible, but break to avoid infinite loop
                break
            
            # Restricted Candidate List (RCL)
            # Select blocks with gain close to max_gain
            # alpha controls randomness. 0.5 is a good start.
            alpha = 0.5
            threshold = max_gain * (1 - alpha)
            
            # Find candidates
            candidates = np.where((gains >= threshold) & (gains > 0) & (~used_block_mask))[0]
            
            if len(candidates) == 0:
                # Fallback to best
                best_idx = np.argmax(gains)
            else:
                # Randomly pick from candidates
                best_idx = random.choice(candidates)
                
            # Add block
            blocks_used_indices.append(best_idx)
            used_block_mask[best_idx] = True
            uncovered[block_t_indices[best_idx]] = 0
            
        # Local Search: Deletion
        # Convert indices to list of blocks
        blocks = [all_blocks[i] for i in blocks_used_indices]
        
        # Build coverage counts
        t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
        # Map block tuple to its index in all_blocks for fast lookup
        # We already have block_to_idx
        
        # Update counts
        for b in blocks:
            idx = block_to_idx[b]
            t_sub_count[block_t_indices[idx]] += 1
            
        # Deletion Phase
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle pop safely? Or just mark and filter.
            # For efficiency, we can just scan and remove.
            # Since we might remove many, restarting is safer for correctness of counts.
            # But O(N^2) worst case.
            # Let's do a single pass first, then repeat.
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                idx = block_to_idx[b]
                idxs = block_t_indices[idx]
                
                # Check if any t-subset in this block has count 1
                # If min count > 1, block is redundant
                if np.min(t_sub_count[idxs]) > 1:
                    # Remove block
                    for idx_t in idxs:
                        t_sub_count[idx_t] -= 1
                    blocks.pop(i)
                    changed = True
                    # We can break and restart to ensure no other blocks become redundant 
                    # due to this removal? No, removal only decreases counts, 
                    # so it cannot make other blocks redundant. It can only make 
                    # previously redundant blocks *less* redundant (count drops).
                    # Wait, if I remove A, counts drop. If B was redundant (count>1), 
                    # it might become critical (count=1) if it shared a subset with A.
                    # So we MUST restart or re-evaluate.
                    # Breaking and restarting the while loop is correct.
                    break
                    
        return blocks

    # Run GRASP iterations until time limit
    while time.time() - start_time < time_limit:
        # Check if we already have a perfect solution? No, we don't know the optimum.
        # Just run iterations.
        current_blocks = grasp_iteration()
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
        # Optional: Early stop if we hit a very low number? 
        # Theoretical lower bound is 264. If we hit 265, it's excellent.
        # But we don't know if we can do better. Keep going.
        
    if best_blocks is None:
        # Fallback: Return empty or raise error? 
        # The task says "Valid when...". If we fail, score 0.
        # But GRASP should always produce a valid covering.
        return []
        
    return [list(b) for b in best_blocks]
