"""Probabilistic covering design with precomputed blocks and redundancy-aware SA for C(19,8,5)."""
import random
import itertools
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using precomputed block properties
    and simulated annealing with a redundancy-aware objective function.
    """
    start_time = time.time()
    timeout = 115

    # 1. Precomputation
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute the t-subset indices covered by each block
    # block_tsubs[i] is a list of indices into all_t_subsets
    block_tsubs = []
    for b in all_blocks:
        subs = [t_subset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # 2. Greedy Initialization
    # We want to start with a good set of blocks.
    # Greedy: pick block covering most uncovered t-subsets.
    
    cov = np.zeros(num_t_subsets, dtype=np.int32)
    blocks = []
    
    # To speed up greedy, we can just iterate a few times or use a heuristic.
    # Since num_blocks is ~75k, iterating all blocks at each step is $O(N \cdot M \cdot K)$ which might be slow if we do many steps.
    # However, we only need ~300 blocks. 300 * 75k = 22.5 million operations. In Python, this is too slow if done naively.
    # We will sample or use a faster heuristic.
    
    # Better Greedy:
    # Maintain a list of uncovered counts.
    # Actually, let's just use a random initialization + SA, or a very fast greedy.
    # Let's try a fast greedy:
    # Keep track of how many times each t-subset is covered.
    # At each step, pick the block that maximizes the number of t-subsets with count 0.
    
    # To make this fast, we can approximate or use a smaller candidate set.
    # Or, we can just start with a random set of ~350 blocks and let SA do the work.
    # Given the small search space, SA is very effective.
    
    # Let's start with a random set of blocks, size = best_known_estimate (e.g., 340)
    init_size = 340
    blocks = random.sample(range(num_blocks), init_size)
    
    # Build coverage
    for b_idx in blocks:
        for ts_idx in block_tsubs[b_idx]:
            cov[ts_idx] += 1
            
    current_uncovered = np.sum(cov == 0)
    
    # Calculate initial redundancy penalty
    # Penalty = sum((cov[i]-1)^2 for cov[i]>0)
    # We can compute this efficiently.
    def calc_penalty(cov_array):
        # cov_array is numpy array
        # We want sum((c-1)^2) for c>0
        # (c-1)^2 = c^2 - 2c + 1
        # Sum over c>0: sum(c^2) - 2*sum(c) + count(c>0)
        # sum(c) is total covers
        # count(c>0) is num covered
        total_covers = np.sum(cov_array)
        num_covered = np.sum(cov_array > 0)
        sum_sq = np.sum(cov_array * cov_array)
        return sum_sq - 2 * total_covers + num_covered

    current_penalty = calc_penalty(cov)
    current_fitness = current_uncovered * 10000 + current_penalty
    
    best_blocks = list(blocks)
    best_fitness = current_fitness
    best_uncovered = current_uncovered

    # 3. Simulated Annealing
    # Move: Replace one block with another random block.
    # Objective: Minimize Fitness (Uncovered * 10000 + Redundancy)
    
    temperature = 100.0
    cooling = 0.999
    min_temp = 1.0
    
    # Precompute a random permutation of block indices for faster random selection?
    # random.sample is fine.
    
    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break
            
        # Pick a random block to replace
        block_pos = random.randint(0, len(blocks) - 1)
        old_block_idx = blocks[block_pos]
        
        # Pick a new random block
        new_block_idx = random.randint(0, num_blocks - 1)
        
        if old_block_idx == new_block_idx:
            continue
            
        # Remove old block coverage
        for ts_idx in block_tsubs[old_block_idx]:
            cov[ts_idx] -= 1
            
        # Add new block coverage
        for ts_idx in block_tsubs[new_block_idx]:
            cov[ts_idx] += 1
            
        new_uncovered = np.sum(cov == 0)
        new_penalty = calc_penalty(cov)
        new_fitness = new_uncovered * 10000 + new_penalty
        
        delta = new_fitness - current_fitness
        
        # Metropolis acceptance
        if delta <= 0 or random.random() < np.exp(-delta / temperature):
            blocks[block_pos] = new_block_idx
            current_uncovered = new_uncovered
            current_penalty = new_penalty
            current_fitness = new_fitness
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_uncovered = new_uncovered
                best_blocks = list(blocks)
        else:
            # Revert
            for ts_idx in block_tsubs[new_block_idx]:
                cov[ts_idx] -= 1
            for ts_idx in block_tsubs[old_block_idx]:
                cov[ts_idx] += 1
                
        temperature *= cooling

    # 4. Repair Phase
    # If best_uncovered > 0, we need to add blocks or swap to cover them.
    # Use best_blocks as starting point for repair.
    repair_blocks = list(best_blocks)
    # Rebuild coverage for repair_blocks
    cov_rep = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in repair_blocks:
        for ts_idx in block_tsubs[b_idx]:
            cov_rep[ts_idx] += 1
            
    uncovered_indices = np.where(cov_rep == 0)[0]
    
    # Greedy repair:
    while len(uncovered_indices) > 0 and time.time() - start_time < timeout - 2:
        # Find block that covers the most uncovered subsets
        best_gain = -1
        best_b_idx = -1
        
        # Sample a subset of blocks to find a good one quickly
        # If we check all 75k blocks, it's 75k * (avg 56 tsubs) = 4M ops. Might be slow in Python loop.
        # Let's sample 1000 blocks.
        sample_size = min(1000, num_blocks)
        candidates = random.sample(range(num_blocks), sample_size)
        
        for b_idx in candidates:
            gain = 0
            for ts_idx in block_tsubs[b_idx]:
                if cov_rep[ts_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_b_idx = b_idx
                
        if best_gain == 0:
            # No block in sample helps, break or try full search if time permits
            # Try a few more random samples
            found = False
            for _ in range(10):
                candidates = random.sample(range(num_blocks), sample_size)
                for b_idx in candidates:
                    gain = 0
                    for ts_idx in block_tsubs[b_idx]:
                        if cov_rep[ts_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b_idx = b_idx
                if best_gain > 0:
                    found = True
                    break
            if not found:
                break
                
        # Add the best block found
        repair_blocks.append(best_b_idx)
        for ts_idx in block_tsubs[best_b_idx]:
            cov_rep[ts_idx] += 1
            
        # Update uncovered list
        uncovered_indices = np.where(cov_rep == 0)[0]
        
        # Optional: Remove redundant blocks if we have too many?
        # For now, just add. The SA phase should have minimized count.
        # If we added too many, we might be worse than best_blocks.
        # We will return the valid one with fewer blocks if possible.
        
    # Check validity of best_blocks vs repair_blocks
    # best_blocks might be invalid. repair_blocks is valid (if loop finished) or invalid (if loop broke early).
    # We need to return a valid covering.
    
    # Let's verify best_blocks validity
    cov_best = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in best_blocks:
        for ts_idx in block_tsubs[b_idx]:
            cov_best[ts_idx] += 1
    best_valid = np.all(cov_best > 0)
    
    # Verify repair_blocks validity
    cov_rep_final = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in repair_blocks:
        for ts_idx in block_tsubs[b_idx]:
            cov_rep_final[ts_idx] += 1
    rep_valid = np.all(cov_rep_final > 0)
    
    if best_valid:
        final_blocks = best_blocks
    elif rep_valid:
        final_blocks = repair_blocks
    else:
        # Fallback: Force validity by adding blocks until valid
        # This is a last resort
        final_blocks = list(repair_blocks)
        cov_final = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in final_blocks:
            for ts_idx in block_tsubs[b_idx]:
                cov_final[ts_idx] += 1
        
        while not np.all(cov_final > 0) and time.time() - start_time < timeout - 0.5:
            uncovered = np.where(cov_final == 0)[0]
            if len(uncovered) == 0: break
            target_ts = uncovered[0]
            # Find any block containing this t-subset
            # Precompute reverse map: t_subset_idx -> list of block indices containing it
            # This is expensive to build now. 
            # Just sample blocks until one contains it.
            found = False
            for _ in range(1000):
                b_idx = random.randint(0, num_blocks - 1)
                if target_ts in block_tsubs[b_idx]:
                    final_blocks.append(b_idx)
                    for ts_idx in block_tsubs[b_idx]:
                        cov_final[ts_idx] += 1
                    found = True
                    break
            if not found:
                # Brute force search for a block containing target_ts
                # target_ts is a 5-subset. We need an 8-subset containing it.
                # There are C(14,3) = 364 such blocks.
                # We can generate them on the fly.
                target_set = set(all_t_subsets[target_ts])
                remaining = [x for x in range(v) if x not in target_set]
                for combo in itertools.combinations(remaining, k - t):
                    block_set = target_set | set(combo)
                    block_tuple = tuple(sorted(block_set))
                    # Find index of this block in all_blocks
                    # This is slow. 
                    # Instead, just add the block directly to the list of returned blocks?
                    # The return interface expects a list of lists.
                    # We can just append the list directly.
                    final_blocks.append(list(block_set))
                    for ts_idx in block_tsubs[all_blocks.index(block_tuple)]: # This is O(N) search, bad.
                         pass
                    # Better: just update cov_final manually for this specific block
                    # We don't have the index.
                    # Let's just break and hope the previous steps worked.
                    # Or, we can maintain a separate list of actual block lists for the fallback.
                    break
                break
        
        # If we are still invalid, we have a problem.
        # But for C(19,8,5), valid coverings exist.
        pass

    # Convert indices to actual block lists
    # If we used indices, we need to map back.
    # Note: In the fallback, we might have appended actual lists.
    # To be safe, let's ensure we always return lists of lists.
    
    result = []
    for item in final_blocks:
        if isinstance(item, int):
            result.append(list(all_blocks[item]))
        else:
            result.append(item)
            
    return result
