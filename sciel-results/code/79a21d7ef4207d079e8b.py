"""Stronger covering design algorithm for C(v, k, t)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

import numpy as np


def construct(v, k, t):
    """
    Greedy covering design with multiple strategies and a time budget.
    
    Strategy:
    1. Fast greedy: pick the t-subset with the fewest candidate blocks, then
       greedily build a block that covers many uncovered t-subsets.
    2. Local search polish: try to remove redundant blocks.
    3. Multiple restarts with different random seeds if time permits.
    """
    start_time = time.time()
    time_limit = 115  # seconds, leave margin before 120s kill
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def time_elapsed():
        return time.time() - start_time
    
    def build_greedy_block(uncovered_set, block_seed, rng):
        """
        Build a k-block starting from block_seed (a t-subset) that covers
        as many uncovered t-subsets as possible.
        """
        block = set(block_seed)
        remaining = k - t
        
        # Precompute which t-subsets are covered by current block
        def covered_count(block_set):
            count = 0
            for ts in itertools.combinations(sorted(block_set), t):
                if ts in uncovered_set:
                    count += 1
            return count
        
        for _ in range(remaining):
            best_point = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in block]
            
            # Shuffle for tie-breaking randomness
            rng.shuffle(candidates)
            
            for x in candidates:
                new_block = block | {x}
                # Count newly covered t-subsets
                gain = 0
                for rest in itertools.combinations(sorted(new_block - {x}), t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            block.add(best_point)
        
        return sorted(block)
    
    def greedy_cover(uncovered_set, rng, use_priority=False):
        """
        Greedy set cover: repeatedly pick an uncovered t-subset, build a block
        around it, and remove all t-subsets covered by that block.
        
        If use_priority, pick the uncovered t-subset that is "hardest" to cover
        (fewest candidate blocks that could cover it).
        """
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered:
            if time_elapsed() > time_limit:
                break
            
            if use_priority:
                # Pick the uncovered t-subset with the fewest possible covering blocks
                # Approximate: count how many points are in other uncovered subsets
                # Simpler heuristic: just pick randomly for speed
                ts = rng.choice(list(uncovered))
            else:
                ts = min(uncovered)
            
            # Build a block centered on this t-subset
            block = build_greedy_block(uncovered, ts, rng)
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for covered_ts in itertools.combinations(block, t):
                uncovered.discard(covered_ts)
        
        return blocks
    
    def try_remove_redundant(blocks):
        """Try to remove blocks that are redundant (all their t-subsets are covered by other blocks)."""
        if not blocks:
            return blocks
        
        n = len(blocks)
        # For each block, check if it can be removed
        # Build a coverage count: for each t-subset, how many blocks cover it
        coverage_count = defaultdict(int)
        block_t_subsets = []
        for block in blocks:
            ts_list = list(itertools.combinations(block, t))
            block_t_subsets.append(ts_list)
            for ts in ts_list:
                coverage_count[ts] += 1
        
        new_blocks = []
        for i, block in enumerate(blocks):
            if all(coverage_count[ts] > 1 for ts in block_t_subsets[i]):
                # This block is redundant
                for ts in block_t_subsets[i]:
                    coverage_count[ts] -= 1
            else:
                new_blocks.append(block)
        
        return new_blocks
    
    # Run multiple greedy strategies and keep the best
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Strategy 1: Standard greedy
    blocks1 = greedy_cover(set(all_t_subsets), rng, use_priority=False)
    if len(blocks1) < best_count:
        best_count = len(blocks1)
        best_blocks = blocks1
    
    # Strategy 2: Greedy with random selection
    if time_elapsed() < time_limit - 30:
        rng2 = random.Random(123)
        blocks2 = greedy_cover(set(all_t_subsets), rng2, use_priority=True)
        if len(blocks2) < best_count:
            best_count = len(blocks2)
            best_blocks = blocks2
    
    # Strategy 3: Try local search polish on the best solution
    if best_blocks and time_elapsed() < time_limit - 20:
        polished = try_remove_redundant(best_blocks)
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = polished
    
    # More restarts if time permits
    if best_blocks and time_elapsed() < time_limit - 40:
        for seed in [7, 99, 2024]:
            if time_elapsed() > time_limit - 30:
                break
            rng3 = random.Random(seed)
            blocks3 = greedy_cover(set(all_t_subsets), rng3, use_priority=False)
            if len(blocks3) < best_count:
                best_count = len(blocks3)
                best_blocks = blocks3
            # Try polish
            if time_elapsed() < time_limit - 25:
                polished3 = try_remove_redundant(blocks3)
                if len(polished3) < best_count:
                    best_count = len(polished3)
                    best_blocks = polished3
    
    # Final check: ensure we have a valid cover
    if best_blocks is None:
        # Fallback: simple greedy
        rng4 = random.Random(0)
        best_blocks = greedy_cover(set(all_t_subsets), rng4, use_priority=False)
    
    # Verify coverage
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    # If not fully covered (due to time), extend with additional blocks
    if len(covered) < n_t_subsets:
        uncovered = set(all_t_subsets) - covered
        while uncovered:
            if time_elapsed() > time_limit:
                break
            ts = min(uncovered)
            block = build_greedy_block(uncovered, ts, random.Random())
            best_blocks.append(block)
            for covered_ts in itertools.combinations(block, t):
                uncovered.discard(covered_ts)
    
    return best_blocks
# EVOLVE-BLOCK-END
