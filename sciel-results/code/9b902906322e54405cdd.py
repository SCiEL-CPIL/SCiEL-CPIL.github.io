"""Improved program for covering design C(19,7,4) using bitmask-based vectorized greedy."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin from 150s limit

    # Precompute all t-subsets as tuples
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Create a mapping from t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each possible block (k-subset), which t-subsets it covers
    # Represent as a bitmask using Python integers (arbitrary precision)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_subsets)
    
    # Precompute coverage masks: for each block, a set of t-subset indices it covers
    # Use a list of frozensets for fast lookup
    block_coverage = []
    for block in all_k_subsets:
        covered = set()
        for t_sub in itertools.combinations(block, t):
            covered.add(subset_to_idx[t_sub])
        block_coverage.append(covered)
    
    # Convert to a numpy boolean matrix for vectorized operations
    # coverage_matrix[i, j] = 1 if block i covers t-subset j
    coverage_matrix = np.zeros((n_blocks, n_subsets), dtype=np.bool_)
    for i, covered in enumerate(block_coverage):
        for j in covered:
            coverage_matrix[i, j] = True
    
    # Greedy set cover with vectorized gain computation
    def greedy_set_cover(seed_offset=0):
        if time.time() > deadline:
            return None
        
        uncovered = np.ones(n_subsets, dtype=np.bool_)
        blocks_used = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Compute gain for each block: number of currently uncovered t-subsets it covers
            gains = coverage_matrix[:, uncovered].sum(axis=1)
            
            # Find the block with maximum gain (with random tie-breaking based on seed)
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            if seed_offset > 0 and len(candidates) > 1:
                rng = random.Random(seed_offset + len(blocks_used))
                chosen = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                chosen = candidates[0]
            
            blocks_used.append(all_k_subsets[chosen])
            uncovered[coverage_matrix[chosen]] = False
        
        if not uncovered.any():
            return blocks_used
        return None
    
    # Try greedy with different seeds
    best_blocks = None
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        result = greedy_set_cover(seed)
        if result is not None:
            if best_blocks is None or len(result) < len(best_blocks):
                best_blocks = result
    
    # Local search: try to remove redundant blocks
    if best_blocks is not None and time.time() < deadline - 3:
        best_blocks = local_search_refine(best_blocks, v, k, t, subset_to_idx, deadline)
    
    if best_blocks is None:
        # Fallback: simple greedy without vectorization
        best_blocks = fallback_greedy(v, k, t, subset_to_idx, deadline)
    
    return best_blocks


def local_search_refine(blocks, v, k, t, subset_to_idx, deadline):
    """Try to remove redundant blocks from the solution."""
    if not blocks:
        return blocks
    
    # Build coverage map: for each t-subset, which blocks cover it
    # Use a dict: t_subset -> list of block indices
    coverage_count = {}
    for idx, block in enumerate(blocks):
        for t_sub in itertools.combinations(block, t):
            if t_sub not in coverage_count:
                coverage_count[t_sub] = []
            coverage_count[t_sub].append(idx)
    
    # Try to remove each block if it's redundant
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        current_blocks = list(blocks)
        n = len(current_blocks)
        
        for i in range(n - 1, -1, -1):
            if time.time() > deadline - 1:
                break
            
            # Check if block i is redundant
            redundant = True
            for t_sub in itertools.combinations(current_blocks[i], t):
                covers = coverage_count.get(t_sub, [])
                # Count how many blocks (other than i) cover this t_sub
                count = sum(1 for j in covers if j != i)
                if count == 0:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                current_blocks.pop(i)
                # Update coverage counts
                for t_sub in itertools.combinations(current_blocks[i], t):
                    if t_sub in coverage_count:
                        coverage_count[t_sub] = [j for j in coverage_count[t_sub] if j > i]
                        # Actually need to be more careful with index shifting
                        # Let's rebuild coverage counts after each removal
                        coverage_count = {}
                        for idx, block in enumerate(current_blocks):
                            for ts in itertools.combinations(block, t):
                                if ts not in coverage_count:
                                    coverage_count[ts] = []
                                coverage_count[ts].append(idx)
                        break
                improved = True
    
    return current_blocks


def fallback_greedy(v, k, t, subset_to_idx, deadline):
    """Simple greedy fallback if vectorized approach fails."""
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered_set = set(all_t_subsets)
    blocks = []
    
    while uncovered_set and time.time() < deadline:
        base_subset = min(uncovered_set)
        current_block = list(base_subset)
        current_block_set = set(current_block)
        
        while len(current_block) < k:
            if time.time() > deadline:
                break
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in current_block_set:
                    continue
                gain = 0
                for combo in itertools.combinations(current_block, t - 1):
                    t_sub = tuple(sorted(combo + (x,)))
                    if t_sub in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                break
            
            current_block.append(best_point)
            current_block_set.add(best_point)
        
        if len(current_block) < k:
            remaining = [x for x in range(v) if x not in current_block_set]
            for x in remaining[:k - len(current_block)]:
                current_block.append(x)
                current_block_set.add(x)
        
        current_block.sort()
        blocks.append(current_block)
        
        for t_sub in itertools.combinations(current_block, t):
            uncovered_set.discard(t_sub)
    
    return blocks
# EVOLVE-BLOCK-END
