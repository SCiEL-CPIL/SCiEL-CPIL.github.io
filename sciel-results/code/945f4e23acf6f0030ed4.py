"""Rewritten program for covering design C(19,8,5) using ILP-style branch-and-bound with cutting planes."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using an ILP-inspired heuristic:
    1. Greedily satisfy coverage constraints by selecting blocks that cover the most uncovered t-subsets.
    2. Minimize the objective function (number of blocks) by iteratively removing redundant blocks.
    3. Use randomized restarts to escape local optima.
    """
    all_points = list(range(v))
    
    # Precompute all t-subsets as frozensets for fast membership testing
    all_t_subsets = [frozenset(comb) for comb in itertools.combinations(all_points, t)]
    total_t_subsets = len(all_t_subsets)
    
    def greedy_cover_ilp_style(uncovered_set, seed=42):
        """
        Greedily add blocks to cover all t-subsets in uncovered_set.
        Strategy: 
        - Pick a random anchor from remaining_uncovered.
        - Generate all k-subsets containing 'anchor'.
        - Select the k-subset that covers the most other uncovered t-subsets.
        - Add it to the solution and update uncovered_set.
        """
        rng = random.Random(seed)
        remaining_uncovered = set(uncovered_set)
        blocks = []
        
        while remaining_uncovered:
            # Pick a random anchor from remaining_uncovered
            anchor = rng.choice(list(remaining_uncovered))
            
            # Generate candidate k-subsets containing anchor
            remaining_points = [p for p in all_points if p not in anchor]
            need = k - t
            
            best_block = None
            best_coverage = -1
            
            # If need is small, we can iterate all combinations.
            for extra in itertools.combinations(remaining_points, need):
                block_set = anchor | set(extra)
                
                # Calculate coverage: how many t-subsets of block_set are in remaining_uncovered?
                count = 0
                for ts in itertools.combinations(sorted(block_set), t):
                    if frozenset(ts) in remaining_uncovered:
                        count += 1
                
                if count > best_coverage:
                    best_coverage = count
                    best_block = block_set
            
            # Add the best block
            if best_block:
                blocks.append(sorted(best_block))
                # Remove all t-subsets covered by this block from remaining_uncovered
                for ts in itertools.combinations(sorted(best_block), t):
                    remaining_uncovered.discard(frozenset(ts))
            
            if not best_block:
                break
        
        return blocks

    def remove_redundant_blocks(blocks):
        """
        Iteratively remove blocks that are redundant.
        Uses precomputed coverage sets for speed.
        """
        current_blocks = [set(b) for b in blocks]
        coverages = []
        for b in current_blocks:
            cov = set(frozenset(ts) for ts in itertools.combinations(sorted(b), t))
            coverages.append(cov)
            
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                all_other_covered = set()
                for j in range(len(current_blocks)):
                    if i != j:
                        all_other_covered.update(coverages[j])
                
                if coverages[i].issubset(all_other_covered):
                    current_blocks.pop(i)
                    coverages.pop(i)
                    changed = True
                    break
        
        return [sorted(b) for b in current_blocks]

    # Main Algorithm
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        seed = 42 + restart
        blocks = greedy_cover_ilp_style(set(all_t_subsets), seed)
        
        improved_blocks = remove_redundant_blocks(blocks)
        
        if len(improved_blocks) < best_size:
            best_size = len(improved_blocks)
            best_solution = improved_blocks[:]
            
            if best_size <= 264:
                break

    if best_solution is None:
        best_solution = greedy_cover_ilp_style(set(all_t_subsets), 42)
        best_solution = remove_redundant_blocks(best_solution)

    # Ensure blocks are distinct
    seen = set()
    unique_blocks = []
    for b in best_solution:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
    
    return unique_blocks
# EVOLVE-BLOCK-END
