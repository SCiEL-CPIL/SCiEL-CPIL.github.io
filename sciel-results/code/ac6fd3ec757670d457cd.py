import itertools
import numpy as np
from scipy.special import comb
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Covering Density heuristic.
    
    Strategy:
    1. Initialize with a dense, highly symmetric pool of blocks derived from cyclic structures.
    2. Use a fast greedy set cover from this pool to get an initial covering.
    3. Apply fast vectorized local search that minimizes sum of inverse coverage counts.
       - Phase 1: Remove redundant blocks.
       - Phase 2: Swap blocks to reduce energy (sum of 1/coverage).
    """
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # --- Step 1: Generate Candidate Pool ---
    # We generate a pool of blocks based on cyclic shifts of random base blocks.
    # This ensures high symmetry and density.
    random.seed(42)
    
    base_blocks = []
    # Generate a diverse set of base blocks
    # We want blocks that cover many distinct triples.
    # A simple heuristic: pick random blocks, but ensure they are distinct.
    num_bases = 200
    seen_bases = set()
    for _ in range(num_bases):
        base = tuple(sorted(random.sample(range(v), k)))
        if base not in seen_bases:
            seen_bases.add(base)
            base_blocks.append(base)
    
    candidate_blocks = []
    seen_blocks = set()
    for base in base_blocks:
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
    
    # Precompute t-subset indices for each candidate block
    block_t_indices = []
    for block in candidate_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_indices.append(np.array(indices, dtype=np.int32))
    
    # --- Step 2: Initial Greedy Cover ---
    # We want to cover all t-subsets.
    # Greedy: pick the block that covers the most currently uncovered t-subsets.
    
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        best_block_idx = -1
        max_cover = 0
        
        # Iterate through candidates. To speed up, we can skip blocks that are already selected?
        # No, we can select the same block multiple times? No, blocks must be distinct.
        # So we track which are selected.
        
        for i in range(len(candidate_blocks)):
            if i in selected_block_indices:
                continue
            count = np.sum(uncovered[block_t_indices[i]])
            if count > max_cover:
                max_cover = count
                best_block_idx = i
        
        if best_block_idx == -1 or max_cover == 0:
            # Fallback: add random blocks if greedy fails (shouldn't happen if pool is good)
            break
        
        selected_block_indices.append(best_block_idx)
        uncovered[block_t_indices[best_block_idx]] = False
        
    blocks = [list(candidate_blocks[i]) for i in selected_block_indices]
    
    # If not covered, add random blocks
    if uncovered.any():
        while uncovered.any():
            if time.time() - start_time > time_limit:
                break
            block = tuple(sorted(random.sample(range(v), k)))
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            indices = []
            for t_sub in itertools.combinations(block, t):
                indices.append(t_subset_to_idx[t_sub])
            indices_arr = np.array(indices, dtype=np.int32)
            if np.sum(uncovered[indices_arr]) > 0:
                blocks.append(list(block))
                uncovered[indices_arr] = False
                # Add to candidate pool for future swaps? 
                # For simplicity, we'll just add to a separate list or extend candidate_blocks
                # But we need to update block_t_indices too.
                candidate_blocks.append(block)
                block_t_indices.append(indices_arr)
    
    # --- Step 3: Local Search ---
    # Objective: Minimize number of blocks, then minimize energy E = sum(1/coverage_i)
    
    # Current state
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    current_block_t_indices = []
    
    # Re-initialize coverage counts
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for block in blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        indices_arr = np.array(indices, dtype=np.int32)
        current_block_t_indices.append(indices_arr)
        coverage_counts[indices_arr] += 1
    
    # Phase 1: Remove redundant blocks
    # A block is removable if all its t-subsets have coverage >= 2.
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if np.all(coverage_counts[current_block_t_indices[i]] > 1):
                # Remove block i
                coverage_counts[current_block_t_indices[i]] -= 1
                blocks.pop(i)
                current_block_t_indices.pop(i)
                changed = True
                break  # Restart loop to check new indices
    
    # Phase 2: Swap to enable more removals
    # We want to find a block B and a replacement C such that:
    # 1. Swapping B->C maintains coverage (all subsets still covered).
    # 2. The new configuration allows removing more blocks than the old one.
    # This is complex. A simpler proxy:
    # Minimize the "energy" E = sum(1/c_i). Lower energy means coverage is more evenly distributed,
    # which often allows more blocks to be removed (if c_i >= 2 for more subsets).
    
    def compute_energy(counts):
        positive = counts[counts > 0]
        if len(positive) == 0:
            return float('inf')
        return np.sum(1.0 / positive)
    
    current_energy = compute_energy(coverage_counts)
    
    improved = True
    # Limit number of swap iterations to avoid getting stuck
    max_swap_iterations = 2000
    iter_count = 0
    
    while improved and time.time() - start_time < time_limit and iter_count < max_swap_iterations:
        improved = False
        iter_count += 1
        
        # Try to swap each block
        # To be efficient, we don't try all pairs (N*M). We sample candidates.
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            
            # Temporarily remove block i
            temp_counts = coverage_counts.copy()
            temp_counts[current_block_t_indices[i]] -= 1
            
            # Check if coverage is still valid (all > 0)
            # If not, we can't remove it without replacement, but we are swapping, so we need to add one.
            # The swap is valid if (temp_counts + count(C)) has no zeros.
            
            # Sample candidate blocks
            # We prefer candidates that cover subsets with low coverage in temp_counts
            # Let's just sample random candidates from the pool
            sample_size = min(50, len(candidate_blocks))
            # Avoid picking the same block as i? 
            # We need to map current block i to its index in candidate_blocks if it exists.
            # For now, just pick random indices.
            
            best_candidate_idx = -1
            best_new_energy = current_energy
            
            # We need to be careful: if the candidate is the same as the one we are removing, it's not a swap.
            # We can just check if the block content is the same.
            current_block_tuple = tuple(blocks[i])
            
            for _ in range(sample_size):
                cand_idx = random.randint(0, len(candidate_blocks) - 1)
                cand_block = candidate_blocks[cand_idx]
                if cand_block == current_block_tuple:
                    continue
                
                cand_indices = block_t_indices[cand_idx]
                
                # Calculate new counts
                new_counts = temp_counts.copy()
                new_counts[cand_indices] += 1
                
                # Check validity
                if np.any(new_counts == 0):
                    continue
                
                # Calculate new energy
                # Optimization: Only calculate energy for changed indices?
                # Let's compute delta energy.
                old_contrib = 0.0
                new_contrib = 0.0
                
                # Get unique indices affected
                affected = np.union1d(current_block_t_indices[i], cand_indices)
                
                for j in affected:
                    old_contrib += 1.0 / coverage_counts[j]
                    new_contrib += 1.0 / new_counts[j]
                
                new_energy = current_energy - old_contrib + new_contrib
                
                if new_energy < best_new_energy:
                    best_new_energy = new_energy
                    best_candidate_idx = cand_idx
            
            if best_candidate_idx != -1 and best_new_energy < current_energy:
                # Perform swap
                coverage_counts = temp_counts.copy()
                coverage_counts[block_t_indices[best_candidate_idx]] += 1
                blocks[i] = list(candidate_blocks[best_candidate_idx])
                current_block_t_indices[i] = block_t_indices[best_candidate_idx].copy()
                current_energy = best_new_energy
                improved = True
                
                # After a swap, check if we can remove any blocks
                # We break out of the inner loop and restart the outer loop?
                # Or just continue. Removing is handled in Phase 1.
                # Let's do a quick removal check here.
                for j in range(len(blocks)):
                    if np.all(coverage_counts[current_block_t_indices[j]] > 1):
                        coverage_counts[current_block_t_indices[j]] -= 1
                        blocks.pop(j)
                        current_block_t_indices.pop(j)
                        # Adjust i if j < i? 
                        # This is messy. Let's just break and let the main loop handle removals.
                        break
    
    # Final Removal Pass
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if np.all(coverage_counts[current_block_t_indices[i]] > 1):
                coverage_counts[current_block_t_indices[i]] -= 1
                blocks.pop(i)
                current_block_t_indices.pop(i)
                changed = True
                break
    
    # Verify coverage
    final_counts = np.zeros(n_t_subsets, dtype=np.int32)
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            final_counts[t_subset_to_idx[t_sub]] += 1
    
    # If there are uncovered subsets, add blocks to cover them
    if np.any(final_counts == 0):
        uncovered_mask = (final_counts == 0)
        while uncovered_mask.any():
            if time.time() - start_time > time_limit:
                break
            # Find a block that covers the most uncovered subsets
            best_block = None
            best_count = 0
            for _ in range(100):
                block = tuple(sorted(random.sample(range(v), k)))
                count = 0
                for t_sub in itertools.combinations(block, t):
                    if uncovered_mask[t_subset_to_idx[t_sub]]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is not None:
                blocks.append(list(best_block))
                for t_sub in itertools.combinations(best_block, t):
                    final_counts[t_subset_to_idx[t_sub]] += 1
                    uncovered_mask[t_subset_to_idx[t_sub]] = False
    
    return blocks
