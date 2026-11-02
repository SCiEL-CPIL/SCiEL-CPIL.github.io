import itertools
import random
import math
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute all k-subsets and their covered t-subset indices
    # This might be heavy but v=19, k=7 is small enough.
    # C(19, 7) = 50388
    # Each block has C(7, 4) = 35 t-subsets.
    # Total storage: ~1.7M ints.
    
    all_k_subsets = []
    block_t_indices = []
    
    for block in itertools.combinations(range(v), k):
        all_k_subsets.append(block)
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(t_subset_to_index[ts])
        block_t_indices.append(t_indices)
        
    num_k_subsets = len(all_k_subsets)
    
    # 1. Greedy Initialization
    # We want a valid covering.
    # uncovered_counts: count of how many times each t-subset is covered by current blocks.
    # We start with 0 blocks.
    # We need to add blocks until all counts > 0.
    
    # To speed up greedy, we can just pick blocks that cover the most currently uncovered subsets.
    # However, maintaining "uncovered" dynamically is tricky with simple arrays if we don't track which are uncovered.
    # Let's just use a set of uncovered t-subset indices.
    
    uncovered = set(range(num_t_subsets))
    initial_blocks = []
    
    # Greedy loop
    while uncovered:
        best_block_idx = -1
        max_covered = 0
        
        # Optimization: We can't check all 50k blocks every step if we do many steps.
        # But greedy is only done once. 50k * 35 checks is ~1.7M ops. Fast.
        for i in range(num_k_subsets):
            # Count how many of this block's t-subsets are in 'uncovered'
            # This is slow if 'uncovered' is a set and we iterate 35 times.
            # Better: just check intersection size?
            # Or, we can pre-compute nothing, just do it.
            count = 0
            for idx in block_t_indices[i]:
                if idx in uncovered:
                    count += 1
                    # Early break if count exceeds current best?
                    if count > max_covered:
                        # We can't break immediately because we need to check if it's the max
                        # But if we find a block covering ALL uncovered, we can stop.
                        # Unlikely.
                        pass
            
            if count > max_covered:
                max_covered = count
                best_block_idx = i
                
        if best_block_idx == -1:
            break # Should not happen if covering exists
            
        initial_blocks.append(best_block_idx)
        
        # Update uncovered
        for idx in block_t_indices[best_block_idx]:
            uncovered.discard(idx)
            
    # Convert initial_blocks to list of block indices
    current_blocks = initial_blocks[:]
    
    # Calculate initial cost
    # Cost = len(blocks) + lambda * num_uncovered
    # For valid solution, num_uncovered = 0.
    
    # 2. Simulated Annealing
    # State: list of block indices
    # We maintain coverage counts to quickly check validity.
    
    coverage_counts = [0] * num_t_subsets
    for b_idx in current_blocks:
        for ts_idx in block_t_indices[b_idx]:
            coverage_counts[ts_idx] += 1
            
    num_uncovered = 0
    for c in coverage_counts:
        if c == 0:
            num_uncovered += 1
            
    # If initial greedy failed (shouldn't), force validity
    # But greedy is exact for set cover if we iterate all blocks.
    
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    # SA Parameters
    T = 10.0
    cooling = 0.99
    min_T = 0.01
    lambda_penalty = 100.0 # Heavy penalty for invalidity
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a block
            if not current_blocks:
                T *= cooling
                continue
            b_idx_pos = random.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[b_idx_pos]
            
            # Calculate delta
            delta_uncovered = 0
            for ts_idx in block_t_indices[b_idx]:
                if coverage_counts[ts_idx] == 1:
                    delta_uncovered += 1
            
            delta_cost = -1 + lambda_penalty * delta_uncovered
            
            # Acceptance
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Apply move
                current_blocks.pop(b_idx_pos)
                for ts_idx in block_t_indices[b_idx]:
                    coverage_counts[ts_idx] -= 1
                    if coverage_counts[ts_idx] == 0:
                        num_uncovered += 1
                # Note: num_uncovered update above is slightly redundant if we track delta, 
                # but we need to keep num_uncovered consistent for future checks?
                # Actually, we only need num_uncovered for cost calculation if we don't recalc.
                # We calculated delta_uncovered. So num_uncovered += delta_uncovered.
                # Let's stick to updating the array and recalculating num_uncovered if needed, 
                # or just track it.
                # The loop above updates counts. We also updated num_uncovered in the loop?
                # No, I wrote `num_uncovered += 1` inside the loop. That's correct.
                
                if num_uncovered == 0 and len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                    
        elif move_type < 0.7:
            # Add a block
            # Heuristic: Pick a block that covers the most uncovered subsets
            # If no uncovered, pick random to explore?
            # If valid, adding a block increases cost, so we only do it if it helps future removals?
            # In SA, we allow bad moves.
            
            # Find block covering most uncovered
            best_add_idx = -1
            max_cover = 0
            # Sample a subset of blocks to speed up? Or all?
            # 50k blocks * 35 checks = 1.7M. Might be slow per step.
            # Let's sample 1000 random blocks.
            sample_size = min(1000, num_k_subsets)
            candidates = random.sample(range(num_k_subsets), sample_size)
            
            for i in candidates:
                count = 0
                for ts_idx in block_t_indices[i]:
                    if coverage_counts[ts_idx] == 0:
                        count += 1
                if count > max_cover:
                    max_cover = count
                    best_add_idx = i
                    
            if best_add_idx == -1:
                # No uncovered? Or all candidates covered 0?
                # If valid, adding is bad. If invalid, adding might help.
                # Just pick a random block.
                best_add_idx = random.randint(0, num_k_subsets - 1)
                
            b_idx = best_add_idx
            
            # Calculate delta
            delta_uncovered = 0
            for ts_idx in block_t_indices[b_idx]:
                if coverage_counts[ts_idx] == 0:
                    delta_uncovered -= 1 # uncovered decreases
            
            delta_cost = 1 + lambda_penalty * delta_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Apply move
                current_blocks.append(b_idx)
                for ts_idx in block_t_indices[b_idx]:
                    if coverage_counts[ts_idx] == 0:
                        num_uncovered -= 1
                    coverage_counts[ts_idx] += 1
                
                if num_uncovered == 0 and len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                    
        else:
            # Swap a block
            if not current_blocks:
                T *= cooling
                continue
                
            # Pick block to remove
            b_idx_pos = random.randint(0, len(current_blocks) - 1)
            b_idx_old = current_blocks[b_idx_pos]
            
            # Pick block to add
            # Heuristic: similar to add
            best_add_idx = -1
            max_cover = 0
            sample_size = min(1000, num_k_subsets)
            candidates = random.sample(range(num_k_subsets), sample_size)
            
            # For swap, we want a block that covers what b_old covers (to maintain validity)
            # Or just generally good.
            for i in candidates:
                if i == b_idx_old:
                    continue
                count = 0
                for ts_idx in block_t_indices[i]:
                    # Check if it covers any t-subset that ONLY b_old covers?
                    # That's complex. Just count uncovered or total.
                    # Let's count how many of its t-subsets are currently covered by <= 1 block.
                    # If we remove b_old, those become critical.
                    for ts_idx in block_t_indices[b_idx_old]:
                        if ts_idx in block_t_indices[i]: # This is O(35*35) = 1225. Too slow.
                            pass
                    # Simplified: just pick random or max uncovered.
                    # Let's just pick random for now to ensure mixing.
                    pass
                    
            # Let's just pick a random block different from old
            while True:
                b_idx_new = random.randint(0, num_k_subsets - 1)
                if b_idx_new != b_idx_old:
                    break
            
            # Calculate delta
            # Remove old, Add new
            # We need to know if validity is preserved.
            # Delta uncovered is tricky.
            # Let's just apply, check, and revert if bad?
            # Or calculate precisely.
            
            # Precise calculation:
            # 1. Decrement counts for b_old. Note how many become 0.
            # 2. Increment counts for b_new. Note how many become > 0 (from 0).
            
            # We need to know the state of counts BEFORE the swap.
            # Let's simulate.
            
            # Temporarily apply remove
            counts_before = coverage_counts[:] # Copying 3876 ints is fast enough? ~15KB. Yes.
            
            # Apply remove
            for ts_idx in block_t_indices[b_idx_old]:
                coverage_counts[ts_idx] -= 1
                
            # Apply add
            for ts_idx in block_t_indices[b_idx_new]:
                coverage_counts[ts_idx] += 1
                
            # Check validity
            new_num_uncovered = 0
            for c in coverage_counts:
                if c == 0:
                    new_num_uncovered += 1
            
            delta_uncovered = new_num_uncovered - num_uncovered
            delta_cost = lambda_penalty * delta_uncovered # Size constant
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_blocks[b_idx_pos] = b_idx_new
                num_uncovered = new_num_uncovered
                if num_uncovered == 0 and len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
            else:
                # Reject
                coverage_counts = counts_before[:]
                
        T *= cooling

    # Ensure we return a valid solution
    # best_blocks should be valid if we only updated best when num_uncovered == 0.
    # But what if initial greedy was valid and we never found a better valid one?
    # best_blocks is initialized to initial_blocks which is valid.
    
    # Convert indices back to lists of points
    final_blocks = [list(all_k_subsets[i]) for i in best_blocks]
    
    return final_blocks
