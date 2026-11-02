import itertools
import numpy as np
from scipy.special import comb
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Covering Density heuristic.
    
    Strategy:
    1. Initialize with a dense, highly symmetric pool of blocks derived from cyclic structures
       to ensure high initial triple redundancy/density.
    2. Use greedy set cover with energy-based scoring to get initial covering.
    3. Apply fast vectorized local search that minimizes sum of inverse coverage counts,
       explicitly targeting removal of blocks with highest marginal redundancy.
    4. Try swap operations to further reduce block count.
    """
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate candidate blocks using cyclic shifts of base blocks
    random.seed(42)
    
    # Generate diverse base blocks
    base_blocks = []
    
    # Add structured base blocks first (good for symmetry)
    # Try to create base blocks with good spread
    for _ in range(50):
        base = sorted(random.sample(range(v), k))
        base_blocks.append(base)
    
    # Add random base blocks
    for _ in range(300):
        base = sorted(random.sample(range(v), k))
        base_blocks.append(base)
    
    # Generate all cyclic shifts
    candidate_blocks = []
    seen_blocks = set()
    for base in base_blocks:
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
    
    # For each candidate block, compute the t-subset indices it covers
    block_t_indices = []
    for block in candidate_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_indices.append(np.array(indices, dtype=np.int32))
    
    # Greedy set cover with energy-based scoring
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_block_indices = []
    
    # For efficiency, we'll track which candidate blocks are still available
    available = np.ones(len(candidate_blocks), dtype=np.bool_)
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        # Find the block that best reduces the energy function
        # Energy = sum of 1/coverage_count for uncovered subsets
        # We want to maximize coverage of uncovered subsets
        # But also consider redundancy: prefer blocks that cover many unique uncovered subsets
        
        # Vectorized computation of uncovered coverage for each available block
        covers = np.zeros(len(candidate_blocks), dtype=np.int32)
        
        # Process in chunks for memory efficiency
        chunk_size = 500
        for start in range(0, len(candidate_blocks), chunk_size):
            end = min(start + chunk_size, len(candidate_blocks))
            for i in range(start, end):
                if not available[i]:
                    continue
                covers[i] = np.sum(uncovered[block_t_indices[i]])
        
        # Find best block
        best_block_idx = np.argmax(covers)
        max_cover = covers[best_block_idx]
        
        if max_cover == 0:
            break
        
        selected_block_indices.append(best_block_idx)
        available[best_block_idx] = False
        
        # Update uncovered
        uncovered[block_t_indices[best_block_idx]] = False
    
    # Convert to list of blocks
    blocks = [list(candidate_blocks[i]) for i in selected_block_indices]
    
    # If we didn't get a complete covering, add more blocks
    if uncovered.any():
        # Add random blocks until covered
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
    
    # Local search: minimize sum of 1/coverage_count
    # First, compute coverage counts
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    block_t_indices_current = []
    for block in blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        indices_arr = np.array(indices, dtype=np.int32)
        block_t_indices_current.append(indices_arr)
        coverage_counts[indices_arr] += 1
    
    # Energy function: sum of 1/coverage_count for all t-subsets
    # We want to minimize this, but also minimize number of blocks
    
    # Phase 1: Remove redundant blocks (all t-subsets have coverage > 1)
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            # Check if all t-subsets in block i have coverage > 1
            if np.all(coverage_counts[block_t_indices_current[i]] > 1):
                # Remove block i
                coverage_counts[block_t_indices_current[i]] -= 1
                blocks.pop(i)
                block_t_indices_current.pop(i)
                changed = True
                break  # Restart the loop since indices changed
    
    # Phase 2: Try to swap blocks to reduce energy
    # For each block, try replacing it with a candidate block that covers more uncovered-ish subsets
    # Actually, we want to minimize sum of 1/count, so we want to increase counts where they're low
    
    # Compute current energy
    def compute_energy(counts):
        # Only count subsets that are covered
        positive = counts[counts > 0]
        if len(positive) == 0:
            return float('inf')
        return np.sum(1.0 / positive)
    
    current_energy = compute_energy(coverage_counts)
    
    # Try swap operations
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            
            # Try removing block i temporarily
            temp_counts = coverage_counts.copy()
            temp_counts[block_t_indices_current[i]] -= 1
            
            # Find candidate blocks that would improve energy when added
            # We want to find a block that covers subsets with low coverage
            best_candidate = -1
            best_new_energy = current_energy
            
            # Sample some candidates for efficiency
            sample_size = min(200, len(candidate_blocks))
            sample_indices = random.sample(range(len(candidate_blocks)), sample_size)
            
            for cand_idx in sample_indices:
                cand_indices = block_t_indices[cand_idx]
                
                # Skip if this is the same block
                if cand_idx == selected_block_indices[i]:
                    continue
                
                # Compute new energy if we add this block
                new_counts = temp_counts.copy()
                new_counts[cand_indices] += 1
                
                # Check if all subsets are still covered
                if np.any(new_counts == 0):
                    continue
                
                new_energy = compute_energy(new_counts)
                
                if new_energy < best_new_energy:
                    best_new_energy = new_energy
                    best_candidate = cand_idx
            
            # If we found a better candidate, make the swap
            if best_candidate != -1 and best_new_energy < current_energy:
                coverage_counts = temp_counts.copy()
                coverage_counts[block_t_indices[best_candidate]] += 1
                blocks[i] = list(candidate_blocks[best_candidate])
                block_t_indices_current[i] = block_t_indices[best_candidate].copy()
                current_energy = best_new_energy
                improved = True
    
    # Phase 3: Final removal pass
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for i in range(len(blocks)):
            if np.all(coverage_counts[block_t_indices_current[i]] > 1):
                coverage_counts[block_t_indices_current[i]] -= 1
                blocks.pop(i)
                block_t_indices_current.pop(i)
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
