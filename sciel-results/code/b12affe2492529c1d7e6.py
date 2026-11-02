"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys
from collections import defaultdict


def _precompute_tsubset_index(v, t):
    """Precompute all t-subsets and map to indices for fast lookup."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    return all_tsubsets, n_tsubsets, sub_to_idx


def _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Greedy covering algorithm using bitmask representation.
    Returns (blocks, uncovered_set) where uncovered_set is a set of t-subset indices not yet covered.
    """
    # uncovered: set of indices of t-subsets not yet covered
    uncovered = set(range(n_tsubsets))
    blocks = []
    
    # Precompute for each point x, the list of t-subset indices that contain x
    point_to_subsets = [[] for _ in range(v)]
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_to_subsets[x].append(idx)
    
    # For each pair (a, b) where a < b, precompute which t-subsets contain both a and b
    # This helps in computing gains more efficiently
    # Actually, let's use a different approach: for each block being built, track which t-subsets
    # are "partially covered" (i.e., they contain some but not all elements of the block)
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to start the block
        start_idx = rng.choice(list(uncovered))
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        block_set = set(block)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block_set:
                    continue
                
                # Compute gain: how many uncovered t-subsets would be covered if we add x
                # A t-subset S is covered by block ∪ {x} if S ⊆ block ∪ {x} and |S| = t
                # So S must contain x and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            if not best_points:
                break
            
            # Pick randomly among best points for diversity
            chosen = rng.choice(best_points)
            block.append(chosen)
            block_set.add(chosen)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-subsets from uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                uncovered.discard(idx)
    
    return blocks, uncovered


def _remove_redundant_blocks(blocks, v, t, deadline, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
    Returns a new list of blocks with redundant ones removed.
    """
    if not blocks:
        return blocks
    
    # coverage[i] = set of block indices that cover t-subset i
    coverage = [set() for _ in range(n_tsubsets)]
    
    for bidx, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                coverage[idx].add(bidx)
    
    # A block is redundant if for every t-subset it covers, there's another block covering it
    non_redundant = []
    for bidx, block in enumerate(blocks):
        if time.time() > deadline:
            break
        redundant = True
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                if len(coverage[idx]) == 1:
                    # Only this block covers this t-subset
                    redundant = False
                    break
        if not redundant:
            non_redundant.append(block)
    
    return non_redundant


def _local_search_improve(blocks, v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Try to improve the covering by removing blocks that become redundant
    after other blocks are removed, and by trying to replace blocks with better ones.
    """
    if not blocks:
        return blocks
    
    # First pass: remove redundant blocks iteratively
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        new_blocks = []
        for block in blocks:
            # Check if this block is still needed
            needed = False
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx.get(sub)
                if idx is None:
                    continue
                # Check if any other block in blocks (excluding this one) covers this sub
                covered_by_other = False
                for other in blocks:
                    if other == block:
                        continue
                    if sub in other:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    needed = True
                    break
            if needed:
                new_blocks.append(block)
            else:
                improved = True
        
        blocks = new_blocks
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin: stop at 110s to leave buffer for cleanup
    deadline = start_time + 110
    
    # Precompute
    all_tsubsets, n_tsubsets, sub_to_idx = _precompute_tsubset_index(v, t)
    
    # Try multiple random restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    
    # First, do a deterministic greedy run
    rng = random.Random(42)
    blocks, uncovered = _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx)
    if len(uncovered) == 0:
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more random restarts if time permits
    if time.time() < deadline - 20:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            blocks, uncovered = _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx)
            if len(uncovered) == 0 and len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
    
    # Remove redundant blocks
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _remove_redundant_blocks(best_blocks, v, t, deadline, all_tsubsets, n_tsubsets, sub_to_idx)
    
    if best_blocks is None:
        # Fallback: return an empty list or a trivial solution
        return []
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
