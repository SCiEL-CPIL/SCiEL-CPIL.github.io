"""Improved covering design C(19,7,4) using vectorized greedy + local search."""
import time
import itertools
import numpy as np

# Deadline guard: evaluator kills at 150s; use 135s as safety margin
DEADLINE = 135.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def construct(v, k, t):
    """
    Construct a covering: every t-subset of {0..v-1} lies in at least one k-block.
    Strategy:
      1. Build a coverage matrix of all t-subsets.
      2. Greedy: repeatedly pick the point-set that covers the most uncovered t-subsets.
      3. Local search / removal: try to remove redundant blocks.
      4. Try to swap blocks to reduce count (limited by time).
    """
    global _START_TIME
    _START_TIME = time.time()  # Reset on each call

    # Precompute all t-subsets as sorted tuples
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each possible k-block, precompute which t-subsets it covers.
    # But C(v,k) can be large; for v=19,k=7 that's C(19,7)=50388.
    # Instead, we'll compute coverage on the fly using bitmasks or arrays.

    # Represent uncovered as a boolean array of length n_tsubs
    uncovered = np.ones(n_tsubs, dtype=np.uint8)

    blocks = []

    # Precompute for each point, the list of t-subsets containing it? 
    # Better: for a given partial block, compute how many uncovered t-subsets 
    # would be covered by adding a candidate point.

    # For efficiency, precompute for each t-subset its membership as a bitmask over points.
    # Actually, let's use a different approach:
    # For each t-subset, store the tuple. To check if a k-block covers a t-subset,
    # check if t-subset is a subset of the block.

    # Greedy construction:
    # At each step, we need to find a k-subset that maximizes the number of newly covered t-subsets.
    # We do this by:
    #   - Starting from the uncovered t-subset with the smallest index (or first uncovered).
    #   - Building a block around it greedily.

    # More efficient: use the "max coverage" greedy per block.
    # For each candidate block (too many), we can't enumerate all.
    # Instead: pick an uncovered t-subset, then grow a block containing it.

    while True:
        if _remaining() < 5.0:
            break
        if not uncovered.any():
            break

        # Find the first uncovered t-subset
        uncovered_indices = np.nonzero(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        first_uncovered_idx = uncovered_indices[0]
        first_uncovered = all_tsubs[first_uncovered_idx]

        # Start block with these t points
        block = list(first_uncovered)

        # Add k - t more points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if _remaining() < 2.0:
                break
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]

            best_point = None
            best_gain = -1
            for x in candidates:
                # Count how many uncovered t-subsets would be covered by adding x to block
                # A t-subset is covered by block∪{x} if it's a subset of block∪{x}
                # and it's currently uncovered.
                # Efficient: for each uncovered t-subset that contains x and is subset of block∪{x}
                # That's equivalent: t-subset contains x, and t-subset \ {x} ⊆ block
                gain = 0
                for idx in uncovered_indices:
                    ts = all_tsubs[idx]
                    if x in ts:
                        rest = tuple(p for p in ts if p != x)
                        if all(p in block_set for p in rest):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point is not None and best_gain > 0:
                block.append(best_point)
            else:
                # No point adds coverage; just add an arbitrary point
                block.append(candidates[0] if candidates else 0)

        block.sort()
        blocks.append(block)

        # Update uncovered: mark all t-subsets that are subsets of this block as covered
        block_set = set(block)
        for idx in range(n_tsubs):
            if uncovered[idx]:
                ts = all_tsubs[idx]
                if all(p in block_set for p in ts):
                    uncovered[idx] = 0

        # If we've exhausted time, break
        if _remaining() < 3.0:
            break

    # If we couldn't complete within time, we need to ensure validity.
    # If uncovered still has entries, we must add more blocks.
    # Let's do a final pass to cover any remaining uncovered t-subsets.
    if uncovered.any():
        while True:
            if _remaining() < 1.0:
                break
            uncovered_indices = np.nonzero(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            first_uncovered_idx = uncovered_indices[0]
            first_uncovered = all_tsubs[first_uncovered_idx]
            block = list(first_uncovered)
            block_set = set(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                # Just add the first candidate
                block.append(candidates[0])
                block_set.add(block[-1])
            block.sort()
            blocks.append(block)
            for idx in range(n_tsubs):
                if uncovered[idx]:
                    ts = all_tsubs[idx]
                    if all(p in set(block) for p in ts):
                        uncovered[idx] = 0

    # Now try to remove redundant blocks
    # A block is redundant if removing it doesn't uncover any t-subset.
    # For each block, check if every t-subset it covers is also covered by another block.
    n_blocks = len(blocks)
    if n_blocks > 1 and _remaining() > 10.0:
        # For each t-subset, count how many blocks cover it
        # coverage_count[i] = number of blocks that cover t-subset i
        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            bset = set(b)
            for idx in range(n_tsubs):
                ts = all_tsubs[idx]
                if all(p in bset for p in ts):
                    coverage_count[idx] += 1

        # Try to remove blocks that are fully redundant
        removed = []
        for bi, b in enumerate(blocks):
            if _remaining() < 5.0:
                break
            bset = set(b)
            # Check if all t-subsets covered by b are covered by at least one other block
            redundant = True
            for idx in range(n_tsubs):
                ts = all_tsubs[idx]
                if all(p in bset for p in ts):
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
            if redundant:
                # Remove this block
                removed.append(bi)
                for idx in range(n_tsubs):
                    ts = all_tsubs[idx]
                    if all(p in bset for p in ts):
                        coverage_count[idx] -= 1

        if removed:
            blocks = [b for i, b in enumerate(blocks) if i not in set(removed)]

    return blocks
