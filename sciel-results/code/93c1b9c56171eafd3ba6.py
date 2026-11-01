"""
Covering design C(24,7,3) solver.

Strategy:
1. Fast greedy construction using bitset representation for speed.
2. Local search: iteratively try to replace blocks with alternatives that
   reduce redundancy, or remove redundant blocks.
3. Deadline guard to ensure we finish within 120s.
"""

import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    # Map from t-subset tuple to index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets contain it
    # point_to_subsets[x] = list of t-subset indices containing x
    point_to_subsets = [[] for _ in range(v)]
    for idx, sub in enumerate(all_t_subsets):
        for x in sub:
            point_to_subsets[x].append(idx)
    
    # Fast greedy: represent uncovered t-subsets as a set of indices
    uncovered = set(range(n_t_subsets))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick the t-subset with minimum "potential" (heuristic: just pick first uncovered)
        # Better: pick a random uncovered one to break symmetry
        target = min(uncovered)
        target_subset = all_t_subsets[target]
        
        # Start block with the points of target_subset
        block = list(target_subset)
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    idx = t_subset_to_idx.get(new_sub, -1)
                    if idx != -1 and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            uncovered.discard(idx)
    
    # If we timed out during greedy, check if we have a valid cover
    # Otherwise, proceed to local search
    
    # Local search: try to improve by removing redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    
    def is_redundant(block, blocks_list):
        """Check if block's t-subsets are all covered by other blocks."""
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            found = False
            for other in blocks_list:
                if other == block:
                    continue
                if sub in itertools.combinations(other, t):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Faster redundancy check using set of covered t-subset indices
    def build_coverage(blocks_list):
        """For each t-subset index, count how many blocks cover it."""
        coverage = [0] * n_t_subsets
        for block in blocks_list:
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                coverage[idx] += 1
        return coverage
    
    def try_remove_redundant(blocks_list):
        """Try to remove redundant blocks."""
        coverage = build_coverage(blocks_list)
        new_blocks = []
        for block in blocks_list:
            # Check if all t-subsets of this block have coverage > 1
            redundant = True
            for sub in itertools.combinations(block, t):
                idx = t_subset_to_idx[sub]
                if coverage[idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                # Remove this block and update coverage
                for sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[sub]
                    coverage[idx] -= 1
                continue
            new_blocks.append(block)
        return new_blocks
    
    # Iterative local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_blocks = try_remove_redundant(blocks)
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            improved = True
    
    # Additional local search: try swapping out individual points in blocks
    # to find better configurations
    while time.time() < deadline:
        improved = False
        coverage = build_coverage(blocks)
        
        for i, block in enumerate(blocks):
            if time.time() > deadline:
                break
            
            # Try removing one point and adding another
            for pos in range(k):
                old_point = block[pos]
                for new_point in range(v):
                    if new_point in block:
                        continue
                    
                    new_block = block[:]
                    new_block[pos] = new_point
                    new_block.sort()
                    
                    # Check if this swap maintains coverage
                    # The t-subsets involving old_point that are lost must be covered elsewhere
                    # The t-subsets involving new_point that are gained are fine
                    valid = True
                    for sub in itertools.combinations(block, t):
                        if old_point in sub and new_point not in sub:
                            idx = t_subset_to_idx[sub]
                            if coverage[idx] <= 1:
                                valid = False
                                break
                    if valid:
                        # Also check that the new block doesn't create issues
                        # (it can only help coverage)
                        blocks[i] = new_block
                        coverage = build_coverage(blocks)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try removing redundant blocks again
            new_blocks = try_remove_redundant(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                improved = True
    
    # Final cleanup: remove any remaining redundant blocks
    while True:
        new_blocks = try_remove_redundant(blocks)
        if len(new_blocks) == len(blocks):
            break
        blocks = new_blocks
    
    return blocks
