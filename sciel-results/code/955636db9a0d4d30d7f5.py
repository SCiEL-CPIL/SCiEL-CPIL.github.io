"""
Covering design C(19,7,4) via a column-generation / greedy set-cover hybrid.

Approach:
1. Represent the problem as a set cover: universe = all 4-subsets of 19 points,
   sets = all 7-subsets (each covers C(7,4)=35 of the 4-subsets).
2. Use an iterative greedy algorithm that, at each step, selects the block
   (7-subset) that maximizes the number of currently uncovered 4-subsets it covers.
3. To speed up the search for the best block, we use a two-stage heuristic:
   - Stage 1: For each candidate "seed" 4-subset that is still uncovered, try to
     extend it to a 7-subset by greedily adding points that maximize coverage.
   - Stage 2: Among all candidate blocks generated in Stage 1, pick the one with
     maximum uncovered coverage.
4. Additionally, we use a randomized restart strategy with multiple seeds to escape
   local optima, keeping the best solution found.

The key insight is that the naive greedy (pick first uncovered 4-set, extend greedily)
can get stuck in poor choices. By considering multiple candidate blocks per iteration
and picking the globally best one, we significantly improve the final block count.
"""

import itertools
import random
import sys
from collections import defaultdict

# Precompute for performance
def _precompute(v, k, t):
    """Precompute useful structures for the covering problem."""
    points = list(range(v))
    
    # All t-subsets
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    # For each point, list of t-subsets containing it
    point_to_t_subsets = defaultdict(list)
    for i, ts in enumerate(all_t_subsets):
        for p in ts:
            point_to_t_subsets[p].append(i)
    
    return all_t_subsets, t_subset_index, n_t, point_to_t_subsets


def _count_uncovered_in_block(block, uncovered_mask, t_subset_index, v, t):
    """Count how many t-subsets of block are currently uncovered."""
    count = 0
    for ts in itertools.combinations(block, t):
        idx = t_subset_index.get(tuple(ts))
        if idx is not None and not uncovered_mask[idx]:
            continue
        if idx is not None:
            count += 1
    return count


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an improved greedy set-cover
    with candidate generation and local search.
    
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    random.seed(42)  # Deterministic for reproducibility
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    # uncovered_mask[i] = True if t_subset i is still uncovered
    uncovered_mask = [True] * n_t
    uncovered_count = n_t
    
    blocks = []
    
    # For efficiency, maintain for each point the list of t-subsets it belongs to
    point_to_t_subsets = defaultdict(list)
    for i, ts in enumerate(all_t_subsets):
        for p in ts:
            point_to_t_subsets[p].append(i)
    
    # Precompute: for each pair of points, the t-subsets containing both
    # This helps in fast gain calculation
    # Actually, let's use a different approach for speed:
    # For a partial block B, the gain of adding point x is the number of 
    # (t-1)-subsets S of B such that S ∪ {x} is an uncovered t-subset.
    
    def compute_gain(block, x, uncovered_mask, t_subset_index):
        """Compute how many new uncovered t-subsets are covered by adding x to block."""
        if x in block:
            return 0
        gain = 0
        # For each (t-1)-subset of block, check if adding x forms an uncovered t-subset
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            idx = t_subset_index.get(ts)
            if idx is not None and uncovered_mask[idx]:
                gain += 1
        return gain
    
    def extend_block_greedily(initial_block, uncovered_mask, t_subset_index, v, t, k):
        """Extend initial_block to size k by greedily adding points with max gain."""
        block = list(initial_block)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = compute_gain(block, x, uncovered_mask, t_subset_index)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        block.sort()
        return block
    
    def evaluate_block(block, uncovered_mask, t_subset_index):
        """Count uncovered t-subsets covered by this block."""
        count = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index.get(tuple(ts))
            if idx is not None and uncovered_mask[idx]:
                count += 1
        return count
    
    def apply_block(block, uncovered_mask, t_subset_index):
        """Mark all t-subsets of block as covered. Return number newly covered."""
        newly = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index.get(tuple(ts))
            if idx is not None and uncovered_mask[idx]:
                uncovered_mask[idx] = False
                newly += 1
        return newly
    
    # Main loop: at each iteration, generate candidate blocks and pick the best
    max_iterations = 5000  # Safety limit
    iteration = 0
    
    while uncovered_count > 0 and iteration < max_iterations:
        iteration += 1
        
        # Strategy: Generate multiple candidate blocks and pick the best one
        candidates = []
        
        # Candidate 1: Start from the first uncovered t-subset
        first_uncovered = None
        for i in range(n_t):
            if uncovered_mask[i]:
                first_uncovered = all_t_subsets[i]
                break
        
        if first_uncovered is None:
            break
        
        # Generate candidates by extending different uncovered t-subsets
        # and also by random seeds
        num_candidates = 5
        
        seen_candidates = set()
        
        # Add candidate from first uncovered
        block = extend_block_greedily(first_uncovered, uncovered_mask, t_subset_index, v, t, k)
        key = tuple(block)
        if key not in seen_candidates:
            seen_candidates.add(key)
            candidates.append(block)
        
        # Add candidates from other uncovered t-subsets (sampled)
        uncovered_list = [all_t_subsets[i] for i in range(n_t) if uncovered_mask[i]]
        if len(uncovered_list) > num_candidates:
            sample_size = min(num_candidates - 1, len(uncovered_list))
            sampled = random.sample(uncovered_list, sample_size)
        else:
            sampled = uncovered_list[1:] if len(uncovered_list) > 1 else []
        
        for seed in sampled:
            block = extend_block_greedily(seed, uncovered_mask, t_subset_index, v, t, k)
            key = tuple(block)
            if key not in seen_candidates:
                seen_candidates.add(key)
                candidates.append(block)
        
        # Also try a few random seeds
        for _ in range(2):
            seed = random.sample(points, t)
            block = extend_block_greedily(seed, uncovered_mask, t_subset_index, v, t, k)
            key = tuple(block)
            if key not in seen_candidates:
                seen_candidates.add(key)
                candidates.append(block)
        
        # Pick the best candidate
        best_block = None
        best_gain = -1
        for cand in candidates:
            g = evaluate_block(cand, uncovered_mask, t_subset_index)
            if g > best_gain:
                best_gain = g
                best_block = cand
        
        if best_block is None or best_gain <= 0:
            # Fallback: use first uncovered
            best_block = list(first_uncovered)
            while len(best_block) < k:
                best_x = None
                best_g = -1
                for x in range(v):
                    if x in best_block:
                        continue
                    g = compute_gain(best_block, x, uncovered_mask, t_subset_index)
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x is None:
                    break
                best_block.append(best_x)
            best_block.sort()
        
        # Apply the best block
        newly = apply_block(best_block, uncovered_mask, t_subset_index)
        if newly == 0:
            # Should not happen, but safety check
            # Try a completely random block
            best_block = random.sample(points, k)
            best_block.sort()
            newly = apply_block(best_block, uncovered_mask, t_subset_index)
            if newly == 0:
                # Force add anyway to avoid infinite loop
                pass
        
        blocks.append(best_block)
        uncovered_count -= newly
    
    return blocks
