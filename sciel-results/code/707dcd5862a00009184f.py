"""Bitmask-based inverse-indexed greedy set cover with redundancy removal."""
import time
import itertools
import numpy as np

# We keep the same entry point and return type: list of blocks (each a list of ints).


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s kill

    # Precompute all t-subsets as bitmasks (integers)
    all_tsets = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_tsets.append(mask)
    n_tsets = len(all_tsets)

    # Coverage count per tset
    coverage = np.zeros(n_tsets, dtype=np.int32)

    # Inverse index: for each tset, list of blocks that cover it
    # We'll store as list of lists; block index -> list of tset indices it covers
    block_tsets = []  # list of arrays of tset indices

    # Mapping from tset mask to its index
    tset_to_idx = {mask: i for i, mask in enumerate(all_tsets)}

    blocks = []  # list of frozensets of points

    def add_block(block_set):
        """Add a block, update coverage and inverse index."""
        nonlocal coverage, block_tsets, blocks
        bidx = len(blocks)
        blocks.append(block_set)
        # Find all tsets covered by this block
        tset_indices = []
        for comb in itertools.combinations(sorted(block_set), t):
            mask = 0
            for x in comb:
                mask |= (1 << x)
            tset_indices.append(tset_to_idx[mask])
        tset_indices = np.array(tset_indices, dtype=np.int32)
        block_tsets.append(tset_indices)
        coverage[tset_indices] += 1

    def remove_block(bidx):
        """Remove a block, update coverage and inverse index."""
        nonlocal coverage, block_tsets, blocks
        tset_indices = block_tsets[bidx]
        coverage[tset_indices] -= 1
        block_tsets[bidx] = None
        blocks[bidx] = None
        # We'll compact later if needed; for now just mark as removed
        # Actually, let's keep it simple: rebuild at the end

    # Greedy phase: while there are uncovered tsets
    while True:
        if time.time() > deadline:
            break
        # Find uncovered tsets
        uncovered_mask = coverage == 0
        if not np.any(uncovered_mask):
            break

        # Pick the uncovered tset with index 0 (arbitrary) to start the block
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        seed_tset_idx = uncovered_indices[0]
        seed_tset_mask = all_tsets[seed_tset_idx]

        # Build a block containing the t points of the seed tset
        block_points = set()
        for x in range(v):
            if seed_tset_mask & (1 << x):
                block_points.add(x)

        # Now add k - t more points to maximize coverage of uncovered tsets
        while len(block_points) < k:
            if time.time() > deadline:
                break
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block_points:
                    continue
                # Compute gain: number of uncovered tsets that would be covered
                # by adding x to the current block
                new_block = block_points | {x}
                # Count uncovered tsets contained in new_block
                gain = 0
                # Iterate over t-1 subsets of current block, add x
                for comb in itertools.combinations(sorted(block_points), t - 1):
                    mask = 0
                    for y in comb:
                        mask |= (1 << y)
                    mask |= (1 << x)
                    if mask in tset_to_idx:
                        idx = tset_to_idx[mask]
                        if coverage[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block_points.add(best_point)

        block_set = frozenset(block_points)
        add_block(block_set)

    # Remove redundant blocks
    # Rebuild a clean structure
    # First, compact blocks list
    clean_blocks = []
    for b in blocks:
        if b is not None:
            clean_blocks.append(b)

    # Try to remove blocks one by one
    removed = [False] * len(clean_blocks)
    # Recompute coverage from scratch for the clean list
    coverage2 = np.zeros(n_tsets, dtype=np.int32)
    block_tsets2 = []
    for b in clean_blocks:
        tset_indices = []
        for comb in itertools.combinations(sorted(b), t):
            mask = 0
            for x in comb:
                mask |= (1 << x)
            tset_indices.append(tset_to_idx[mask])
        tset_indices = np.array(tset_indices, dtype=np.int32)
        block_tsets2.append(tset_indices)
        coverage2[tset_indices] += 1

    # Try to remove each block if all its tsets are still covered
    for i in range(len(clean_blocks)):
        if time.time() > deadline:
            break
        if removed[i]:
            continue
        tsets_i = block_tsets2[i]
        # Check if all tsets in this block are covered by other blocks
        # i.e., coverage2[t] >= 2 for all t in tsets_i
        if np.all(coverage2[tsets_i] >= 2):
            # Remove block i
            coverage2[tsets_i] -= 1
            removed[i] = True

    final_blocks = [list(b) for i, b in enumerate(clean_blocks) if not removed[i]]

    return final_blocks
