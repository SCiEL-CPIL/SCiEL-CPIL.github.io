# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {}
    for i, sub in enumerate(all_t_subsets):
        subset_to_idx[sub] = i

    # Track uncovered subsets using a bitset-like array
    uncovered = np.ones(n_subsets, dtype=bool)

    def get_t_subsets_in_block(block):
        """Return list of indices of t-subsets contained in block."""
        indices = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx.get(tuple(sub))
            if idx is not None:
                indices.append(idx)
        return indices

    def count_new_coverage(block_so_far, candidate_point):
        """Count how many new t-subsets would be covered by adding candidate_point to block_so_far."""
        if len(block_so_far) < t - 1:
            return 0
        new_count = 0
        for rest in itertools.combinations(block_so_far, t - 1):
            new_sub = tuple(sorted(rest + (candidate_point,)))
            idx = subset_to_idx.get(new_sub)
            if idx is not None and uncovered[idx]:
                new_count += 1
        return new_count

    blocks = []
    
    # Phase 1: Greedy construction
    while True:
        if time.time() > deadline - 5.0:
            break
        
        if not uncovered.any():
            break
            
        # Pick the first uncovered t-subset to seed a new block
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_idx = uncovered_indices[0]
        seed_subset = all_t_subsets[seed_idx]
        
        # Start block with the seed subset
        block = list(seed_subset)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() > deadline - 3.0:
                break
            
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                gain = count_new_coverage(block, x)
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point >= 0:
                block.append(best_point)
            else:
                break
        
        block.sort()
        blocks.append(block)
        
        # Mark all t-subsets in this block as covered
        for idx in get_t_subsets_in_block(block):
            uncovered[idx] = False
    
    if not uncovered.any():
        return blocks
    
    # Phase 2: Local search - try to improve by swapping points in blocks
    # This is a simple refinement: for each block, try replacing one point with another
    # that covers more uncovered subsets
    
    for _ in range(50):  # Limited number of refinement rounds
        if time.time() > deadline - 2.0:
            break
        if not uncovered.any():
            break
        
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 2.0:
                break
            
            block = blocks[bi]
            # Try removing each point and finding a better replacement
            for pi in range(k):
                if time.time() > deadline - 2.0:
                    break
                
                original_point = block[pi]
                reduced_block = block[:pi] + block[pi+1:]
                
                # Find best replacement
                best_point = -1
                best_gain = -1
                
                for x in range(v):
                    if x in reduced_block:
                        continue
                    candidate_block = reduced_block + [x]
                    gain = count_new_coverage(candidate_block, -1)  # This won't work directly
                    
                    # Better: compute coverage of candidate_block directly
                    cov_before = 0
                    for idx in get_t_subsets_in_block(block):
                        if uncovered[idx]:
                            cov_before += 1
                    
                    cov_after = 0
                    for idx in get_t_subsets_in_block(candidate_block):
                        if uncovered[idx]:
                            cov_after += 1
                    
                    if cov_after > best_gain:
                        best_gain = cov_after
                        best_point = x
                
                # Only accept if strictly better
                if best_point >= 0 and best_point != original_point:
                    # Check if this actually helps
                    old_cov = 0
                    for idx in get_t_subsets_in_block(block):
                        if uncovered[idx]:
                            old_cov += 1
                    
                    new_block = reduced_block + [best_point]
                    new_block.sort()
                    
                    new_cov = 0
                    for idx in get_t_subsets_in_block(new_block):
                        if uncovered[idx]:
                            new_cov += 1
                    
                    if new_cov > old_cov:
                        blocks[bi] = new_block
                        improved = True
    
    if not uncovered.any():
        return blocks
    
    # Phase 3: Add more greedy blocks if time permits
    while True:
        if time.time() > deadline - 1.0:
            break
        if not uncovered.any():
            break
        
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_idx = uncovered_indices[0]
        seed_subset = all_t_subsets[seed_idx]
        
        block = list(seed_subset)
        while len(block) < k:
            if time.time() > deadline - 1.0:
                break
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = count_new_coverage(block, x)
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point >= 0:
                block.append(best_point)
            else:
                break
        
        block.sort()
        blocks.append(block)
        for idx in get_t_subsets_in_block(block):
            uncovered[idx] = False
    
    return blocks
# EVOLVE-BLOCK-END
