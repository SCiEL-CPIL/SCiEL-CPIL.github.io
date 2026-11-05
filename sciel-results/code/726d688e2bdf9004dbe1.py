"""
Covering design C(19,8,5) solver using a fast greedy heuristic followed by
block-removal local search, with a hard deadline guard.
"""
import time
import random
import numpy as np
from itertools import combinations

# Hard deadline: evaluator kills at 150s; use 140s safety margin.
DEADLINE = 140.0


def _t_subsets_of_block(block, t):
    """Yield all t-subsets of a given block as sorted tuples."""
    return set(combinations(block, t))


def _count_uncovered_for_point(block, x, uncovered, t):
    """Count how many uncovered t-subsets would be covered by adding x to block."""
    # A t-subset containing x is covered iff the other t-1 elements are in block
    # and the full t-subset is in uncovered.
    count = 0
    for rest in combinations(block, t - 1):
        sub = tuple(sorted(rest + (x,)))
        if sub in uncovered:
            count += 1
    return count


def _greedy_construct(v, k, t, time_limit=DEADLINE):
    """
    Greedy set-cover: repeatedly pick the block that covers the most uncovered t-subsets.
    To speed up, we use a two-phase approach:
      1. Start with a random or structured initial block.
      2. Grow the block by adding the point that covers the most uncovered t-subsets.
      3. Optionally, try a few candidate starting points and keep the best.
    """
    start_time = time.time()
    uncovered = set(combinations(range(v), t))
    blocks = []
    points = list(range(v))

    while uncovered and (time.time() - start_time) < time_limit:
        # Pick the first uncovered t-subset as the seed
        seed = min(uncovered)
        block = list(seed)
        block_set = set(block)

        # Grow the block to size k
        while len(block) < k and (time.time() - start_time) < time_limit:
            best_x = -1
            best_gain = -1
            for x in points:
                if x in block_set:
                    continue
                gain = _count_uncovered_for_point(block, x, uncovered, t)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain <= 0:
                break
            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            # Pad with arbitrary points to reach size k
            remaining = [x for x in points if x not in block_set]
            block.extend(remaining[:k - len(block)])
            block_set.update(remaining[:k - len(block)])

        block.sort()
        blocks.append(block)

        # Remove covered t-subsets
        covered = _t_subsets_of_block(block, t)
        uncovered -= covered

    return blocks


def _try_remove_block(blocks, uncovered_count_by_block, v, k, t, time_limit=DEADLINE):
    """
    Try to remove each block and see if the covering is still valid.
    Returns the reduced list of blocks.
    """
    start_time = time.time()
    new_blocks = []
    for block in blocks:
        if (time.time() - start_time) >= time_limit:
            new_blocks.append(block)
            continue
        # Tentatively remove this block
        # Check if all t-subsets of this block are covered by other blocks
        # This is expensive, so we do a quick check: for each t-subset of block,
        # see if it's in any other block.
        can_remove = True
        for sub in combinations(block, t):
            found = False
            for other in blocks:
                if other == block:
                    continue
                if all(x in other for x in sub):
                    found = True
                    break
            if not found:
                can_remove = False
                break
        if can_remove:
            # Remove it
            continue
        else:
            new_blocks.append(block)
    return new_blocks


def construct(v, k, t):
    start_time = time.time()
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, time_limit=DEADLINE - 30)
    
    if time.time() - start_time >= DEADLINE - 10:
        return blocks
    
    # Phase 2: Local search - try to remove redundant blocks
    # Repeat removal passes until no more blocks can be removed
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 10:
        improved = False
        new_blocks = _try_remove_block(blocks, None, v, k, t, time_limit=DEADLINE - 10)
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            improved = True
    
    # Phase 3: Try to swap blocks for potentially better coverage
    # (Simple heuristic: try replacing a block with a different one that covers
    #  the same t-subsets but allows more removals)
    # This is complex; for now, we rely on the removal pass.
    
    return blocks
