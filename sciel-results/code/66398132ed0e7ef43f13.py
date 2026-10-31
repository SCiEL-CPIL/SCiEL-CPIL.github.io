import numpy as np
import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all k-subsets (blocks) and t-subsets (targets)
    all_blocks = list(itertools.combinations(range(v), k))
    all_targets = list(itertools.combinations(range(v), t))
    
    num_blocks = len(all_blocks)
    num_targets = len(all_targets)
    
    # Create incidence matrix: M[i, j] = 1 if target j is contained in block i
    # Using a dense boolean matrix is feasible: 50388 * 3876 ~ 195M bits ~ 24MB
    M = np.zeros((num_blocks, num_targets), dtype=bool)
    
    # Map target combinations to indices for fast lookup
    target_to_idx = {tup: i for i, tup in enumerate(all_targets)}
    
    for i, block in enumerate(all_blocks):
        # Find all t-subsets of this block
        for sub in itertools.combinations(block, t):
            idx = target_to_idx[sub]
            M[i, idx] = True
    
    # Precompute the number of targets covered by each block (static property)
    block_coverage_count = M.sum(axis=1)
    
    best_blocks = None
    best_count = float('inf')
    
    # 2. Vectorized Greedy Construction with Restarts
    # We run multiple restarts. In each restart, we greedily pick the block that covers
    # the most *currently uncovered* targets.
    
    # To speed up tie-breaking and diversity, we can slightly randomize the initial
    # "priority" or just shuffle the block order if gains are equal.
    # However, pure greedy with vectorized gain is very fast.
    
    # Optimization: Instead of iterating all blocks to find max gain, 
    # we can just compute gains = M[:, uncovered] .sum(axis=1)
    # This is a matrix-vector product (boolean matrix * boolean vector).
    
    while time.time() < deadline - 5.0:
        uncovered = np.ones(num_targets, dtype=bool)
        current_blocks = []
        used_blocks_mask = np.zeros(num_blocks, dtype=bool)
        
        # Optional: Randomly perturb the order of blocks or add noise to gains
        # to escape local minima? Standard greedy is deterministic if gains are unique.
        # We can add small random noise to gains to diversify restarts.
        
        while uncovered.any():
            if time.time() > deadline - 5.0:
                break
                
            # Calculate gain for each unused block:
            # gain[i] = sum(M[i, :] * uncovered)
            # Using np.dot on boolean arrays might be slow if not optimized.
            # M is bool, uncovered is bool. 
            # np.sum(M[:, uncovered], axis=1) creates a temporary slice.
            # Better: np.dot(M, uncovered.astype(np.int8))
            # Or even better: since M is sparse-ish but stored dense, 
            # let's stick to efficient numpy ops.
            
            # gains[i] = number of currently uncovered targets covered by block i
            # We only care about unused blocks, but computing for all is vectorized.
            gains = M.dot(uncovered)  # Result is int array
            
            # Mask out used blocks
            gains[used_blocks_mask] = -1
            
            # Add small random noise to break ties and diversify
            # Noise should be small enough not to change the decision if gain diff > 1
            # But large enough to randomize ties.
            gains += np.random.rand(num_blocks) * 0.1
            
            best_block_idx = np.argmax(gains)
            
            if gains[best_block_idx] <= 0:
                # No block covers any uncovered target? Should not happen if M is complete.
                break
            
            current_blocks.append(all_blocks[best_block_idx])
            used_blocks_mask[best_block_idx] = True
            
            # Update uncovered: targets covered by this block are no longer uncovered
            # uncovered = uncovered & ~M[best_block_idx]
            # M[best_block_idx] is a boolean vector.
            uncovered &= ~M[best_block_idx]
            
            # Early termination check
            if not uncovered.any():
                break
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
            # If we hit a very good number, we might stop earlier?
            # Lower bound is 131. Best known 151.
            if best_count <= 140:
                # Keep searching a bit more to see if we can go lower, 
                # but don't spend all time if we are already very good.
                # Actually, just let the loop run until deadline.
                pass

    # 3. Local Search: Redundancy Removal
    if best_blocks is not None and time.time() < deadline - 2.0:
        # Convert best_blocks to a set of tuples for fast lookup if needed, 
        # but we need to check coverage.
        
        # Rebuild coverage counts for the current best solution
        # coverage_count[j] = number of blocks in best_blocks that cover target j
        
        # We can use the precomputed M.
        # Create a mask of which blocks are in best_blocks
        block_mask = np.zeros(num_blocks, dtype=bool)
        for b in best_blocks:
            # Find index of b in all_blocks
            # This is O(N) lookup per block, which is slow for ~150 blocks.
            # Better: map block tuple to index.
            pass
        
        # Faster: Map block tuples to indices
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        
        active_blocks_mask = np.zeros(num_blocks, dtype=bool)
        for b in best_blocks:
            idx = block_to_idx[b]
            active_blocks_mask[idx] = True
        
        # Calculate coverage count for each target
        # coverage_counts = M[active_blocks_mask].sum(axis=0)
        coverage_counts = M[active_blocks_mask].sum(axis=0)
        
        # Try to remove blocks that are redundant
        # A block is redundant if all targets it covers are covered by at least one other active block.
        # i.e., for block i, for all j where M[i,j]==1, coverage_counts[j] > 1.
        
        # We can check this for all blocks at once?
        # For each active block i, check if min(coverage_counts[M[i,:]]) > 1.
        
        # Let's iterate through active blocks in random order to remove redundancies.
        active_indices = np.where(active_blocks_mask)[0]
        random.shuffle(active_indices)
        
        removed_any = True
        while removed_any and time.time() < deadline - 1.0:
            removed_any = False
            for idx in active_indices:
                if not active_blocks_mask[idx]:
                    continue
                
                # Check if block idx is redundant
                # Get targets covered by this block
                covered_targets = M[idx]
                if not covered_targets.any():
                    continue
                    
                # Check if any of these targets has coverage count == 1
                # If min(coverage_counts[covered_targets]) > 1, it's safe to remove.
                # Note: coverage_counts includes this block.
                # So if min > 1, it means at least 2 blocks cover it, so removing 1 leaves >= 1.
                min_cov = np.min(coverage_counts[covered_targets])
                
                if min_cov > 1:
                    # Remove block
                    active_blocks_mask[idx] = False
                    # Update coverage counts
                    coverage_counts -= M[idx]
                    removed_any = True
                    # We might need to re-shuffle or just continue.
                    # Since we are iterating a shuffled list, we just mark it inactive.
        
        # Reconstruct the block list from active_blocks_mask
        final_blocks = []
        for i, b in enumerate(all_blocks):
            if active_blocks_mask[i]:
                final_blocks.append(b)
        
        if len(final_blocks) < best_count:
            best_blocks = final_blocks
            best_count = len(final_blocks)

    if best_blocks is None:
        # Fallback: Should not happen, but return empty or raise
        return []

    return best_blocks
