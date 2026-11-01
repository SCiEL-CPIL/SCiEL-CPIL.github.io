"""Covering design C(19,8,5) via fast greedy + local search (NumPy/Bitset)."""
import time
import itertools
import numpy as np
from collections import defaultdict

# Precompute all 5-subsets of 19 points as tuples
# There are C(19,5) = 11628 such subsets
# We'll index them for fast lookup

def _precompute(v, t):
    """Precompute all t-subsets and their indices."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    return all_tsubsets, subset_to_idx


def construct(v, k, t):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    all_tsubsets, subset_to_idx = _precompute(v, t)
    n_subsets = len(all_tsubsets)

    # uncovered: boolean array, True if not yet covered
    uncovered = np.ones(n_subsets, dtype=bool)

    def _time_check():
        return (time.time() - start_time) < DEADLINE

    # Greedy construction with efficient gain computation
    blocks = []

    while np.any(uncovered) and _time_check():
        # Start a new block: pick the t-subset with highest "frequency" of uncovered supersets
        # For simplicity, pick the lexicographically first uncovered subset
        first_uncovered_idx = np.argmax(uncovered)
        initial_block = list(all_tsubsets[first_uncovered_idx])

        # Grow the block to size k
        block = list(initial_block)
        while len(block) < k and _time_check():
            # For each candidate point x not in block, compute how many new t-subsets it covers
            # A t-subset is covered by adding x if it contains x and (t-1) elements from block
            # and is currently uncovered
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that contain x and exactly t-1 elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(candidate)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0 or best_gain <= 0:
                # No improvement; just pick any remaining point
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = remaining[0]
                else:
                    break
            block.append(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered: mark all t-subsets of this block as covered
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            uncovered[idx] = False

    # Local search: try to remove blocks that are redundant
    # A block is redundant if every t-subset it covers is also covered by another block
    # We'll iterate: for each block, check if removing it still leaves all t-subsets covered
    if _time_check():
        # Build coverage count: for each t-subset, how many blocks cover it
        coverage_count = np.zeros(n_subsets, dtype=np.int32)
        for blk in blocks:
            for sub in itertools.combinations(blk, t):
                coverage_count[subset_to_idx[sub]] += 1

        # Try removing blocks that have all their t-subsets covered by >= 2 blocks
        changed = True
        while changed and _time_check():
            changed = False
            # Recompute coverage after potential removals
            coverage_count = np.zeros(n_subsets, dtype=np.int32)
            for blk in blocks:
                for sub in itertools.combinations(blk, t):
                    coverage_count[subset_to_idx[sub]] += 1

            for i in range(len(blocks) - 1, -1, -1):
                if not _time_check():
                    break
                # Check if block i is redundant
                blk = blocks[i]
                redundant = True
                for sub in itertools.combinations(blk, t):
                    idx = subset_to_idx[sub]
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True

    # Try swapping: replace a point in a block with another point to improve coverage
    # This is more complex; for now, do a simple pass of block replacement
    if _time_check():
        # Try to find better blocks by re-greedy on uncovered subsets
        # Actually, let's do a second greedy pass on remaining uncovered
        uncovered = np.zeros(n_subsets, dtype=bool)
        for blk in blocks:
            for sub in itertools.combinations(blk, t):
                uncovered[subset_to_idx[sub]] = True
        # uncovered is now True where covered; we need the complement
        still_uncovered = ~uncovered

        # If there are still uncovered subsets (shouldn't happen if greedy was complete),
        # add more blocks
        while np.any(still_uncovered) and _time_check():
            first_idx = np.argmax(still_uncovered)
            initial_block = list(all_tsubsets[first_idx])
            block = list(initial_block)
            while len(block) < k and _time_check():
                best_x = -1
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(candidate)
                        if idx is not None and still_uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = remaining[0]
                    else:
                        break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                still_uncovered[subset_to_idx[sub]] = False

    return blocks
