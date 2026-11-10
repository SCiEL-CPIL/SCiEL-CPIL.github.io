"""Improved greedy covering with vectorized gain and time management."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets as a flat array
    all_tsubs = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    n_tsubs = len(all_tsubs)

    # For each t-subset, we need to know which blocks cover it.
    # We'll track coverage count for each t-subset.
    covered = np.zeros(n_tsubs, dtype=np.int32)

    # Map from t-subset tuple to index for fast lookup
    tsub_to_idx = {}
    for i, sub in enumerate(itertools.combinations(range(v), t)):
        tsub_to_idx[sub] = i

    blocks = []

    # Helper: get indices of t-subsets contained in a given block
    def block_tsub_indices(block):
        """Return array of indices of t-subsets of the given block."""
        idxs = []
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx.get(sub)
            if idx is not None:
                idxs.append(idx)
        return np.array(idxs, dtype=np.int64)

    # Helper: compute gain vector for adding point x to current partial block
    # A t-subset is covered by adding x if it contains x and (t-1) elements from block
    def gain_vector(partial_block, x):
        """Return a boolean mask of t-subsets that would be newly covered by adding x."""
        # t-subsets that contain x and t-1 elements from partial_block
        mask = np.zeros(n_tsubs, dtype=bool)
        for rest in itertools.combinations(partial_block, t - 1):
            # The t-subset is rest + (x,)
            sub = tuple(sorted(rest + (x,)))
            idx = tsub_to_idx.get(sub)
            if idx is not None:
                mask[idx] = True
        return mask

    # Greedy construction
    while True:
        # Check time
        if time.time() > deadline - 5.0:
            break

        # Find first uncovered t-subset
        uncovered_idx = np.where(covered == 0)[0]
        if len(uncovered_idx) == 0:
            break

        # Start block with the first uncovered t-subset's elements
        first_sub = all_tsubs[uncovered_idx[0]]
        block = list(first_sub)

        # Extend block greedily
        while len(block) < k:
            if time.time() > deadline - 5.0:
                break

            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block:
                    continue
                mask = gain_vector(block, x)
                g = np.sum(mask & (covered == 0))
                if g > best_gain:
                    best_gain = g
                    best_x = x

            if best_x == -1:
                break

            block.append(best_x)

        if len(block) < k:
            # Can't complete the block, try a different starting subset
            # Remove the first uncovered subset from consideration for now
            break

        block.sort()

        # Check if this block is already in blocks
        block_tuple = tuple(block)
        if block_tuple in set(tuple(b) for b in blocks):
            break

        blocks.append(block)

        # Update coverage
        idxs = block_tsub_indices(block)
        if len(idxs) > 0:
            covered[idxs] += 1

    # If we ran out of time or couldn't finish, do a final pass to ensure validity
    # by adding more blocks if needed
    while True:
        if time.time() > deadline - 2.0:
            break

        uncovered_idx = np.where(covered == 0)[0]
        if len(uncovered_idx) == 0:
            break

        # Try to build a block covering the most uncovered t-subsets
        first_sub = all_tsubs[uncovered_idx[0]]
        block = list(first_sub)

        while len(block) < k:
            if time.time() > deadline - 2.0:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                mask = gain_vector(block, x)
                g = np.sum(mask & (covered == 0))
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        if len(block) < k:
            break

        block.sort()
        block_tuple = tuple(block)
        if block_tuple in set(tuple(b) for b in blocks):
            break

        blocks.append(block)
        idxs = block_tsub_indices(block)
        if len(idxs) > 0:
            covered[idxs] += 1

    return blocks
# EVOLVE-BLOCK-END
