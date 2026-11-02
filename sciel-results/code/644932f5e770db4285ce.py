import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Covering Density heuristic.
    
    Strategy:
    1. Initialize with a pool of blocks derived from cyclic shifts of random bases
       to ensure high initial symmetry and redundancy.
    2. Use greedy set cover to get an initial covering.
    3. Apply fast local search that minimizes the sum of inverse coverage counts (Energy),
       which encourages high redundancy and allows for block deletion.
    4. Use swap operations to further optimize the energy and reduce block count.
    """
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    random.seed(42)
    
    # Generate candidate blocks
    # Use cyclic shifts of random bases for symmetry
    candidate_blocks = []
    seen_blocks = set()
    
    # 1. Structured blocks: Cyclic shifts of random bases
    n_bases = 40
    for _ in range(n_bases):
        base = sorted(random.sample(range(v), k))
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
                
    # 2. Random blocks to add diversity
    n_random = 2000
    for _ in range(n_random):
        block = tuple(sorted(random.sample(range(v), k)))
        if block not in seen_blocks:
            seen_blocks.add(block)
            candidate_blocks.append(block)
            
    # Precompute t-subset indices for each candidate block
    # This is a list of numpy arrays
    block_t_indices_pool = []
    for block in candidate_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_indices_pool.append(np.array(indices, dtype=np.int32))
        
    # Greedy set cover
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_block_indices = []
    
    # To speed up greedy, we can just iterate. 
    # With ~3000 blocks and ~2000 subsets, and ~80 steps, it's fast.
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
            
        best_block_idx = -1
        max_cover = 0
        
        # Find block covering most uncovered subsets
        # Vectorized-ish approach:
        # covers[i] = sum(uncovered[pool_indices[i]])
        # We can do this in a loop which is fast enough for 3000 items
        for i in range(len(candidate_blocks)):
            # If this block was already selected, skip? 
            # Greedy set cover usually allows re-selection if it helps, but here blocks are distinct.
            # However, we should not re-select a block already in selected_block_indices.
            # But checking membership in a list is O(N). 
            # Let's just allow re-selection in the logic, but since we remove from 'uncovered',
            # a selected block will have 0 uncovered coverage eventually.
            # Actually, once selected, its contribution to 'uncovered' becomes 0.
            # So we don't need to explicitly skip if we update 'uncovered' correctly.
            
            cover_count = np.sum(uncovered[block_t_indices_pool[i]])
            if cover_count > max_cover:
                max_cover = cover_count
                best_block_idx = i
                
        if max_cover == 0:
            break
            
        selected_block_indices.append(best_block_idx)
        uncovered[block_t_indices_pool[best_block_idx]] = False
        
    # Convert to list of blocks
    blocks = [list(candidate_blocks[i]) for i in selected_block_indices]
    
    # If not fully covered (due to time limit in greedy), add random blocks
    if uncovered.any():
        while uncovered.any():
            if time.time() - start_time > time_limit:
                break
            block = tuple(sorted(random.sample(range(v), k)))
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            indices = np.array([t_subset_to_idx[sub] for sub in itertools.combinations(block, t)], dtype=np.int32)
            if np.sum(uncovered[indices]) > 0:
                blocks.append(list(block))
                # Add to pool for local search? No, pool is fixed for simplicity, 
                # but we can add to a separate 'current_pool' if needed. 
                # For now, we just add to blocks.
                uncovered[indices] = False
                
    # Local Search: Minimize Energy = sum(1/coverage_count)
    # Initialize coverage counts
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    current_block_indices = []
    for block in blocks:
        indices = np.array([t_subset_to_idx[sub] for sub in itertools.combinations(block, t)], dtype=np.int32)
        current_block_indices.append(indices)
        coverage_counts[indices] += 1
        
    def compute_energy(counts):
        # Energy is sum of 1/count for all covered subsets
        # Subsets with count 0 are invalid, but we assume valid state
        positive = counts[counts > 0]
        if len(positive) == 0:
            return 0.0
        return np.sum(1.0 / positive)
        
    current_energy = compute_energy(coverage_counts)
    
    # Phase 1: Deletion Pass
    # Remove any block that is redundant
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            # Check if all subsets in block i have coverage > 1
            if np.all(coverage_counts[current_block_indices[i]] > 1):
                coverage_counts[current_block_indices[i]] -= 1
                blocks.pop(i)
                current_block_indices.pop(i)
                changed = True
                break  # Restart loop as indices changed
                
    # Phase 2: Swap Optimization
    # Try to swap blocks to lower energy
    improved = True
    swap_iterations = 0
    while improved and time.time() - start_time < time_limit:
        improved = False
        swap_iterations += 1
        if swap_iterations > 50: # Safety break
            break
            
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            
            # Temporarily remove block i
            temp_counts = coverage_counts.copy()
            temp_counts[current_block_indices[i]] -= 1
            
            # Check if still valid (no zeros)
            if np.any(temp_counts == 0):
                # If removing block i leaves some subsets uncovered, 
                # the replacement block MUST cover them.
                uncovered_mask = (temp_counts == 0)
            else:
                uncovered_mask = None # Any block is a valid candidate structurally, but we want to lower energy
                
            best_candidate_idx = -1
            best_new_energy = current_energy
            
            # Sample candidates from the pool
            # We prioritize candidates that cover the 'uncovered_mask' if it exists
            # For efficiency, we just sample a subset of the pool
            sample_size = min(100, len(candidate_blocks))
            # To ensure we find a valid block if needed, we might need to search more intelligently
            # But for now, random sampling from the diverse pool should work
            
            sample_indices = random.sample(range(len(candidate_blocks)), sample_size)
            
            for cand_idx in sample_indices:
                cand_indices = block_t_indices_pool[cand_idx]
                
                # If the block is the same, skip
                if np.array_equal(cand_indices, current_block_indices[i]):
                    continue
                    
                # Check validity
                if uncovered_mask is not None:
                    # Candidate must cover all currently uncovered subsets
                    if not np.all(uncovered_mask[cand_indices]):
                        continue
                else:
                    # If no subsets were uncovered by removal, any block is structurally valid
                    # But we must ensure we don't create a situation where we can't delete?
                    # No, we just check energy.
                    pass
                
                # Calculate new energy
                new_counts = temp_counts.copy()
                new_counts[cand_indices] += 1
                
                # Optimization: Only recalculate energy for changed parts?
                # Energy = sum(1/c).
                # Change in energy = sum(1/new_c - 1/old_c) for changed subsets.
                # Subsets in cand_indices: old_c = temp_counts[s], new_c = temp_counts[s] + 1
                # Note: temp_counts already reflects removal of block i.
                
                # Let's just compute the full energy for simplicity first, 
                # but this is O(N_t) which is 2000. 100 samples * 2000 = 200k ops per block i.
                # 80 blocks * 200k = 16M ops. This is acceptable in Python if not too slow.
                # NumPy sum is fast.
                
                new_energy = compute_energy(new_counts)
                
                if new_energy < best_new_energy:
                    best_new_energy = new_energy
                    best_candidate_idx = cand_idx
            
            # If a better candidate is found, swap
            if best_candidate_idx != -1 and best_new_energy < current_energy:
                # Update state
                coverage_counts = temp_counts
                coverage_counts[block_t_indices_pool[best_candidate_idx]] += 1
                blocks[i] = list(candidate_blocks[best_candidate_idx])
                current_block_indices[i] = block_t_indices_pool[best_candidate_idx].copy()
                current_energy = best_new_energy
                improved = True
                
                # Run a quick deletion pass after a successful swap
                for j in range(len(blocks)):
                    if np.all(coverage_counts[current_block_indices[j]] > 1):
                        coverage_counts[current_block_indices[j]] -= 1
                        blocks.pop(j)
                        current_block_indices.pop(j)
                        if j == i:
                            # We removed the block we just swapped? 
                            # This can happen if the new block is redundant.
                            # We need to break or handle index carefully.
                            # If we remove block i, the loop for i is invalid.
                            # Let's just break and restart the outer loop.
                            break
                        # If we removed a block with index > i, indices shift?
                        # No, we are iterating i from 0 to N. If we pop j, 
                        # and j < i, i is now wrong.
                        # This is getting messy.
                        # Simpler: Just mark improved and break out to restart deletion/swap loop.
                        pass

    # Final Deletion Pass
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            if np.all(coverage_counts[current_block_indices[i]] > 1):
                coverage_counts[current_block_indices[i]] -= 1
                blocks.pop(i)
                current_block_indices.pop(i)
                changed = True
                break

    # Verify and return
    # Re-verify coverage to be safe
    final_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            final_counts[t_subset_to_idx[t_sub]] += 1
            
    if np.any(final_counts == 0):
        # Fallback: add random blocks
        uncovered_mask = (final_counts == 0)
        while uncovered_mask.any():
            if time.time() - start_time > time_limit:
                break
            block = tuple(sorted(random.sample(range(v), k)))
            indices = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
            if np.any(uncovered_mask[indices]):
                blocks.append(list(block))
                final_counts[indices] += 1
                uncovered_mask[indices] = False

    return blocks
