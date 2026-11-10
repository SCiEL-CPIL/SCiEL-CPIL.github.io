import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    """
    Build a covering of all t-subsets of v points using k-subsets (blocks).
    Strategy:
      1. Fast greedy construction using bitset-based uncovered tracking.
      2. Local search: try to remove redundant blocks or replace blocks with better ones.
    """
    start = time.time()
    deadline = start + 110.0  # 10s safety margin before 120s kill

    # Precompute all t-subsets of {0..v-1}
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # Track uncovered t-subsets as a numpy boolean array
    uncovered = np.ones(n_tsubs, dtype=bool)

    def tsub_indices_for_block(block):
        """Return the indices of t-subsets covered by a given block."""
        idxs = []
        for combo in itertools.combinations(block, t):
            idxs.append(tsub_index[combo])
        return np.array(idxs, dtype=np.int64)

    blocks = []

    # Phase 1: Greedy construction
    while True:
        if time.time() > deadline - 30.0:
            break  # leave time for local search
        if not uncovered.any():
            break

        # Pick the uncovered t-subset with smallest "frequency" among remaining candidates
        # For simplicity and speed, pick the first uncovered one (lexicographically)
        uncovered_indices = np.nonzero(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_tsub = all_tsubs[uncovered_indices[0]]
        block = list(seed_tsub)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                new_block = block + [x]
                idxs = tsub_indices_for_block(new_block)
                gain = np.count_nonzero(uncovered[idxs])
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        blocks.append(block)
        # Update uncovered
        idxs = tsub_indices_for_block(block)
        uncovered[idxs] = False

    # Phase 2: Local search - try to reduce number of blocks
    if time.time() > deadline - 5.0:
        return blocks

    random.seed(42)

    # Try to remove redundant blocks
    for iteration in range(500):
        if time.time() > deadline - 2.0:
            break
        if not blocks:
            break

        # Pick a random block index
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]

        # Check if this block is redundant (all its t-subsets are covered by other blocks)
        idxs_old = tsub_indices_for_block(old_block)
        covered_without = np.zeros(n_tsubs, dtype=bool)
        for j, b in enumerate(blocks):
            if j == i:
                continue
            idxs = tsub_indices_for_block(b)
            covered_without[idxs] = True

        # Check if all t-subsets in old_block are covered by other blocks
        if np.all(covered_without[idxs_old]):
            # Block is redundant, remove it
            blocks.pop(i)

    # Try random replacements
    for iteration in range(200):
        if time.time() > deadline - 1.0:
            break
        if not blocks:
            break

        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]

        # Generate a random new block
        new_block = sorted(random.sample(range(v), k))
        if new_block == old_block:
            continue

        # Check if replacing old_block with new_block maintains coverage
        idxs_old = tsub_indices_for_block(old_block)
        idxs_new = tsub_indices_for_block(new_block)

        # Build coverage without old_block
        covered_without = np.zeros(n_tsubs, dtype=bool)
        for j, b in enumerate(blocks):
            if j == i:
                continue
            idxs = tsub_indices_for_block(b)
            covered_without[idxs] = True

        # Check if new_block covers all t-subsets that old_block uniquely covered
        if np.all(covered_without[idxs_old]):
            # Safe to replace
            blocks[i] = new_block

    return blocks
