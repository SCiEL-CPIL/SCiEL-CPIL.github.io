"""
Covering design C(v,k,t) solver using randomized local search with
bitset-based coverage tracking and k-for-(k+1) exchange moves.

Strategy:
1. Build a fast greedy initial covering using numpy bitsets for coverage.
2. Iteratively try to remove blocks that are "redundant" (their t-subsets
   are all covered by other blocks). If removal leaves coverage intact,
   accept it.
3. Additionally, try swap moves: remove one block, add another (randomly
   generated or greedily chosen) if it reduces the count or maintains it
   while enabling further reductions.
4. Use a deadline guard to stop before timeout.
"""

import time
import random
import numpy as np
import itertools
from collections import defaultdict

def _build_tsubset_to_index(v, t):
    """Map each t-subset (as a frozenset or tuple) to a unique index."""
    subs = list(itertools.combinations(range(v), t))
    mapping = {tuple(s): i for i, s in enumerate(subs)}
    return subs, mapping

def _greedy_build(v, k, t, rng, time_limit, subs, mapping):
    """
    Greedy set cover: repeatedly pick the block that covers the most
    currently-uncovered t-subsets.
    """
    n_tsubsets = len(subs)
    # coverage_count[i] = how many blocks cover t-subset i
    coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
    blocks = []
    
    deadline = time.time() + time_limit
    uncovered_mask = np.ones(n_tsubsets, dtype=bool)
    
    while uncovered_mask.any():
        if time.time() > deadline:
            break
        
        # For each candidate point set, compute how many uncovered t-subsets it covers
        # We'll do a smarter greedy: start with the first uncovered t-subset,
        # then greedily add points that maximize new coverage.
        
        # Pick the first uncovered t-subset as seed
        idx = np.argmax(uncovered_mask)  # first True
        seed = list(subs[idx])
        
        # Build block greedily
        block = list(seed)
        
        while len(block) < k:
            best_x = -1
            best_gain = 0
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset {a1,...,a_{t-1}, x} is covered if {a1,...,a_{t-1}} ⊆ block
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms an uncovered t-subset
                for rest in itertools.combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    if tsub in mapping:
                        i = mapping[tsub]
                        if uncovered_mask[i]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update coverage
        for tsub in itertools.combinations(block, t):
            i = mapping[tuple(tsub)]
            coverage_count[i] += 1
            if coverage_count[i] == 1:
                uncovered_mask[i] = False
    
    return blocks, coverage_count, uncovered_mask

def _try_remove_block(blocks, coverage_count, block_idx, subs, mapping):
    """
    Try to remove blocks[block_idx]. Returns True if removal is safe
    (all t-subsets of that block are still covered by other blocks).
    """
    block = blocks[block_idx]
    # Check each t-subset of this block: is it covered by another block?
    for tsub in itertools.combinations(block, t):
        i = mapping[tuple(tsub)]
        if coverage_count[i] <= 1:
            return False  # This t-subset is only covered by this block
    return True

def _remove_block(blocks, coverage_count, block_idx, subs, mapping):
    """Actually remove block and update coverage_count."""
    block = blocks[block_idx]
    for tsub in itertools.combinations(block, t):
        i = mapping[tuple(tsub)]
        coverage_count[i] -= 1
    del blocks[block_idx]

def _add_block(blocks, coverage_count, block, mapping):
    """Add a block and update coverage_count."""
    blocks.append(tuple(block))
    for tsub in itertools.combinations(block, t):
        i = mapping[tuple(tsub)]
        coverage_count[i] += 1

def _random_block(v, k, rng):
    return tuple(sorted(rng.sample(range(v), k)))

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    WALL_LIMIT = 150  # seconds
    SAFETY_MARGIN = 10  # seconds to reserve
    
    # Build t-subset index
    subs, mapping = _build_tsubset_to_index(v, t)
    n_tsubsets = len(subs)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy initial construction
    greedy_time = min(30.0, WALL_LIMIT * 0.25)
    blocks, coverage_count, uncovered_mask = _greedy_build(
        v, k, t, rng, greedy_time, subs, mapping
    )
    
    # Verify coverage is complete (should be if greedy finished)
    if uncovered_mask.any():
        # If greedy didn't finish, do a quick completion
        deadline_complete = start_time + WALL_LIMIT - SAFETY_MARGIN
        while uncovered_mask.any() and time.time() < deadline_complete:
            idx = np.argmax(uncovered_mask)
            seed = list(subs[idx])
            block = list(seed)
            while len(block) < k:
                best_x = -1
                best_gain = 0
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tsub = tuple(sorted(rest + (x,)))
                        if tsub in mapping:
                            i = mapping[tsub]
                            if uncovered_mask[i]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            for tsub in itertools.combinations(block, t):
                i = mapping[tuple(tsub)]
                coverage_count[i] += 1
                if coverage_count[i] == 1:
                    uncovered_mask[i] = False
    
    best_count = len(blocks)
    
    # Phase 2: Local search - try to remove redundant blocks
    deadline = start_time + WALL_LIMIT - SAFETY_MARGIN
    no_improve_streak = 0
    max_streak = 200
    
    while time.time() < deadline and no_improve_streak < max_streak:
        improved = False
        # Shuffle order of attempts
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            if idx >= len(blocks):
                continue
            if _try_remove_block(blocks, coverage_count, idx, subs, mapping):
                _remove_block(blocks, coverage_count, idx, subs, mapping)
                improved = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
        
        if improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
    
    # Phase 3: Try swap moves - remove one block, add a different one
    # This can help escape local optima where no single block is removable
    # but a swap enables further reductions
    deadline = start_time + WALL_LIMIT - SAFETY_MARGIN
    swap_attempts = 0
    max_swap_attempts = 5000
    
    while time.time() < deadline and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        if not blocks:
            break
        
        # Pick a random block to potentially replace
        idx = rng.randint(0, len(blocks) - 1)
        
        # Try removing it
        if not _try_remove_block(blocks, coverage_count, idx, subs, mapping):
            # Can't remove directly. Try: temporarily remove, then add a
            # block that covers the newly-uncovered t-subsets
            # This is expensive, so only do it occasionally
            if swap_attempts % 10 != 0:
                continue
            
            # Temporarily remove
            old_block = blocks[idx]
            newly_uncovered = []
            for tsub in itertools.combinations(old_block, t):
                i = mapping[tuple(tsub)]
                if coverage_count[i] <= 1:
                    newly_uncovered.append(i)
                    coverage_count[i] -= 1
            del blocks[idx]
            
            # Try to find a single block that covers all newly_uncovered
            # Greedily build a block
            if newly_uncovered:
                # Start with the first newly uncovered t-subset
                seed = list(subs[newly_uncovered[0]])
                block = list(seed)
                while len(block) < k:
                    best_x = -1
                    best_gain = 0
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            tsub = tuple(sorted(rest + (x,)))
                            if tsub in mapping:
                                i = mapping[tsub]
                                if i in newly_uncovered and coverage_count[i] == 0:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        break
                    block.append(best_x)
                block.sort()
                
                # Check if this block covers all newly_uncovered
                covers_all = True
                for i in newly_uncovered:
                    tsub = subs[i]
                    if not all(p in block for p in tsub):
                        covers_all = False
                        break
                
                if covers_all:
                    _add_block(blocks, coverage_count, block, mapping)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                else:
                    # Rollback: re-add old block
                    _add_block(blocks, coverage_count, list(old_block), mapping)
            else:
                # No newly uncovered, just remove
                if len(blocks) < best_count:
                    best_count = len(blocks)
    
    # Convert to list of lists
    return [list(b) for b in blocks]
