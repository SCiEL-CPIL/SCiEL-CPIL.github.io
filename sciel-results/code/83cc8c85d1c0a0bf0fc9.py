import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Parameters for C(19, 7, 4)
    # v=19, k=7, t=4
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(points, k))
    # Map each k-block to the indices of the t-subsets it covers
    block_t_subset_indices = []
    for block in all_k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[s] for s in subs]
        block_t_subset_indices.append(indices)
        
    block_t_subset_indices = np.array(block_t_subset_indices, dtype=np.int32)
    num_k_blocks = len(all_k_blocks)
    
    # State
    subset_counts = np.zeros(num_t_subsets, dtype=np.int32)
    selected_blocks = []
    
    # Greedy Phase
    # We want to pick blocks that cover the most currently uncovered t-subsets.
    # uncovered_mask = (subset_counts == 0)
    # scores = sum(uncovered_mask[block_t_subset_indices[i]] for i in range(num_k_blocks))
    
    while time.time() < deadline:
        # Check if all covered
        if np.all(subset_counts > 0):
            break
            
        # Calculate scores for all blocks
        # subset_counts[block_t_subset_indices] is shape (num_k_blocks, t_subsets_per_block)
        # We want to count how many of these have count 0.
        is_uncovered = (subset_counts[block_t_subset_indices] == 0)
        scores = np.sum(is_uncovered, axis=1)
        
        max_score = np.max(scores)
        if max_score == 0:
            break # Should not happen if not all covered, but safety
            
        # Pick a block with max score. Randomize among ties?
        max_indices = np.where(scores == max_score)[0]
        chosen_block_idx = max_indices[np.random.randint(len(max_indices))]
        
        selected_blocks.append(chosen_block_idx)
        subset_counts[block_t_subset_indices[chosen_block_idx]] += 1
        
    # Initial Redundancy Removal
    def remove_redundants(selected_blocks, subset_counts):
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid order bias
            random.shuffle(selected_blocks)
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                # Temporarily remove
                subset_counts[block_t_subset_indices[b_idx]] -= 1
                if np.any(subset_counts == 0):
                    # Restore
                    subset_counts[block_t_subset_indices[b_idx]] += 1
                else:
                    # Remove permanently
                    selected_blocks.pop(i)
                    changed = True
        return selected_blocks, subset_counts

    selected_blocks, subset_counts = remove_redundants(selected_blocks, subset_counts)
    
    # Local Search Phase
    # Try swaps and add/remove to improve.
    # Objective: Minimize len(selected_blocks).
    # Since swaps don't change size, we use them to find states where redundancy removal yields smaller size.
    
    best_blocks = list(selected_blocks)
    best_size = len(best_blocks)
    
    # Helper to check validity and get new counts
    def try_swap(b_old_idx, b_new_idx, current_counts):
        # Returns (is_valid, new_counts)
        # We don't want to modify current_counts in place for checking
        # But copying 3876 ints is fast.
        new_counts = current_counts.copy()
        new_counts[block_t_subset_indices[b_old_idx]] -= 1
        new_counts[block_t_subset_indices[b_new_idx]] += 1
        if np.any(new_counts == 0):
            return False, None
        return True, new_counts

    # Precompute neighbors for each block?
    # There are 50388 blocks. For each, 7*12=84 neighbors.
    # Total 4M neighbors. Storing them might be memory heavy but doable?
    # 4M * (7 bytes) ~ 28MB. OK.
    # But generating them takes time.
    # Let's generate on the fly or precompute a list of neighbor indices for each block.
    
    block_to_points = {i: b for i, b in enumerate(all_k_blocks)}
    point_to_blocks = {p: [] for p in range(v)}
    for i, b in enumerate(all_k_blocks):
        for p in b:
            point_to_blocks[p].append(i)
            
    # For a block B, neighbors are blocks that differ by exactly 1 point.
    # i.e., B' = (B \ {x}) U {y} where x in B, y not in B.
    # We can precompute neighbors for the blocks we actually use?
    # Or just generate on the fly.
    
    def get_neighbors(block_idx):
        b = all_k_blocks[block_idx]
        neighbors = []
        b_set = set(b)
        for x in b:
            for y in range(v):
                if y not in b_set:
                    new_b = tuple(sorted((b_set - {x}) | {y}))
                    # Find index of new_b
                    # This is slow. 
                    # Better: Precompute a map from block tuple to index.
                    pass
        return neighbors

    # Precompute block_tuple_to_idx
    block_tuple_to_idx = {b: i for i, b in enumerate(all_k_blocks)}
    
    def get_neighbors_fast(block_idx):
        b = all_k_blocks[block_idx]
        b_set = set(b)
        neighbors = []
        for x in b:
            for y in range(v):
                if y not in b_set:
                    new_b = tuple(sorted((b_set - {x}) | {y}))
                    if new_b in block_tuple_to_idx:
                        neighbors.append(block_tuple_to_idx[new_b])
        return neighbors

    # Local Search Loop
    current_blocks = list(selected_blocks)
    current_counts = subset_counts.copy()
    
    # We will iterate, trying to improve.
    # Each step:
    # 1. Try a random swap. If valid, accept.
    # 2. Every N swaps, try redundancy removal.
    # 3. If size improves, update best.
    # 4. If stuck, perturb (add random block, remove redundants).
    
    swap_count = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb: Add a random block
            rand_block = np.random.randint(num_k_blocks)
            if rand_block not in current_blocks:
                current_blocks.append(rand_block)
                current_counts[block_t_subset_indices[rand_block]] += 1
                # Remove redundants
                current_blocks, current_counts = remove_redundants(current_blocks, current_counts)
                no_improve_count = 0
                # If size is still worse, we are in a bad local minimum.
                # Maybe revert to best?
                if len(current_blocks) > best_size:
                    current_blocks = list(best_blocks)
                    current_counts = np.zeros(num_t_subsets, dtype=np.int32)
                    for b in current_blocks:
                        current_counts[block_t_subset_indices[b]] += 1
            continue

        # Pick a random block to swap
        if not current_blocks:
            break
            
        b_old_idx = current_blocks[np.random.randint(len(current_blocks))]
        neighbors = get_neighbors_fast(b_old_idx)
        
        if not neighbors:
            continue
            
        b_new_idx = neighbors[np.random.randint(len(neighbors))]
        
        # Check validity
        valid, new_counts = try_swap(b_old_idx, b_new_idx, current_counts)
        
        if valid:
            # Accept swap
            current_blocks[current_blocks.index(b_old_idx)] = b_new_idx
            current_counts = new_counts
            swap_count += 1
            no_improve_count += 1
            
            # Periodically check for redundancy removal
            if swap_count % 50 == 0:
                new_blocks, new_counts = remove_redundants(current_blocks, current_counts)
                if len(new_blocks) < len(current_blocks):
                    current_blocks = new_blocks
                    current_counts = new_counts
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                        no_improve_count = 0
                else:
                    # No size reduction, keep current state but counts might have changed?
                    # remove_redundants only removes, so counts are consistent with new_blocks.
                    # If no blocks removed, new_blocks is same, new_counts same.
                    pass
        else:
            no_improve_count += 1
            
    return [list(all_k_blocks[i]) for i in best_blocks]
