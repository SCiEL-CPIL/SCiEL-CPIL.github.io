"""Genetic algorithm approach for covering design C(19,8,5)."""
import itertools
import random
import numpy as np
from collections import Counter
import time

def construct(v, k, t):
    """
    Construct a covering design using a hybrid approach:
    1. Start with a good greedy solution
    2. Use local search / genetic-like operations to improve
    """
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each k-subset, the list of t-subsets it covers
    # This is expensive for v=19, k=8, t=5 but we can be smart
    # Instead, we'll work with bitmasks or just compute on the fly
    
    # For efficiency with v=19, k=8, t=5:
    # C(19,5) = 11628 t-subsets
    # C(19,8) = 75582 k-subsets
    # Each k-subset covers C(8,5) = 56 t-subsets
    
    # Greedy initialization
    def greedy_cover():
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        # Precompute coverage for all possible k-subsets? Too memory intensive.
        # Instead, compute coverage on the fly.
        
        while uncovered:
            # Pick a t-subset to cover - choose one that appears in fewest remaining options
            # For simplicity, pick the first uncovered one
            target = min(uncovered)
            target_ts = all_t_subsets[target]
            
            # Find best block containing target_ts
            best_block = None
            best_count = -1
            
            # Generate all k-subsets containing target_ts
            remaining_points = [p for p in range(v) if p not in target_ts]
            need = k - t
            
            for combo in itertools.combinations(remaining_points, need):
                block = tuple(sorted(target_ts + tuple(combo)))
                # Count how many uncovered t-subsets this block covers
                count = 0
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = list(block)
            
            blocks.append(best_block)
            # Remove covered t-subsets
            for ts in itertools.combinations(best_block, t):
                idx = t_subset_to_idx[ts]
                uncovered.discard(idx)
        
        return blocks
    
    # Get initial greedy solution
    blocks = greedy_cover()
    
    # Local search: try to remove redundant blocks
    def is_redundant(blocks, block_idx):
        """Check if block at block_idx is redundant (all its t-subsets covered by other blocks)."""
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            # Check if any other block covers this t-subset
            found = False
            for j, other_block in enumerate(blocks):
                if j == block_idx:
                    continue
                if all(p in other_block for p in ts):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if is_redundant(blocks, i):
                blocks.pop(i)
                changed = True
                break
    
    # Local search: try replacing blocks to improve coverage
    # For each block, try swapping with a better one
    def compute_coverage(blocks):
        """Return set of covered t-subset indices."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                covered.add(idx)
        return covered
    
    # Try to find a better covering through local moves
    current_blocks = list(blocks)
    current_covered = compute_coverage(current_blocks)
    
    # Mutation: for each block, try replacing it with a block that covers more uncovered subsets
    for iteration in range(50):  # Limit iterations for time
        improved = False
        uncovered = set(range(num_t_subsets)) - current_covered
        
        if not uncovered:
            break
        
        for i in range(len(current_blocks)):
            old_block = current_blocks[i]
            old_covered = set()
            for ts in itertools.combinations(old_block, t):
                old_covered.add(t_subset_to_idx[ts])
            
            # New uncovered if we remove this block
            new_uncovered = uncovered | old_covered
            
            # Try to find a replacement block that covers more of new_uncovered
            # Pick a t-subset from new_uncovered and build a block around it
            target_idx = min(new_uncovered)
            target_ts = all_t_subsets[target_idx]
            
            remaining_points = [p for p in range(v) if p not in target_ts]
            need = k - t
            
            best_block = None
            best_gain = 0
            
            for combo in itertools.combinations(remaining_points, need):
                block = tuple(sorted(target_ts + tuple(combo)))
                gain = 0
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if idx in new_uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
            
            if best_block is not None and best_gain > 0:
                # Check if this replacement improves things
                new_blocks = current_blocks.copy()
                new_blocks[i] = best_block
                new_covered = compute_coverage(new_blocks)
                
                if len(new_covered) > len(current_covered):
                    current_blocks = new_blocks
                    current_covered = new_covered
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_blocks
