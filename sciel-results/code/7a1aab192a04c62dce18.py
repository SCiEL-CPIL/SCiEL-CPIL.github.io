"""Stronger greedy covering for C(v,k,t): bitmask + NumPy vectorized gain + deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import time

import numpy as np

_DEADLINE = 150.0
_MARGIN = 3.0


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    combos = list(itertools.combinations(range(v), t))
    n_sub = len(combos)

    # mask of each t-subset (bit over v points)
    masks = np.array([sum(1 << x for x in c) for c in combos], dtype=np.uint16)
    mask_to_idx = {int(m): i for i, m in enumerate(masks)}

    # for each point x, indices of t-subsets containing x (as int array)
    point_subsets = []
    for x in range(v):
        idxs = [i for i, c in enumerate(combos) if x in c]
        point_subsets.append(np.array(idxs, dtype=np.int64))

    # uncovered boolean over t-subsets
    uncovered = np.ones(n_sub, dtype=bool)

    blocks = []

    while True:
        if time.time() > deadline:
            break
        if not uncovered.any():
            break

        first_idx = int(np.argmax(uncovered))
        seed = list(combos[first_idx])

        block = list(seed)
        in_block_mask = sum(1 << b for b in block)

        # t-subset indices already "covered" by the partial block
        partial_covered = np.zeros(n_sub, dtype=bool)
        for c in itertools.combinations(block, t):
            m = sum(1 << x for x in c)
            idx = mask_to_idx.get(m)
            if idx is not None:
                partial_covered[idx] = True

        while len(block) < k:
            candidates = [x for x in range(v) if not (in_block_mask >> x) & 1]
            if not candidates:
                break

            # For each candidate, gain = # uncovered t-subsets containing it
            # that are NOT already covered by the partial block.
            # Vectorize over candidates: build a (n_candidates x n_sub) membership?
            # Too big. Instead: for each candidate, use its point_subsets index list,
            # gather uncovered & ~partial_covered over those indices, and count.
            best_gain = -1
            best_x = None
            for x in candidates:
                idxs = point_subsets[x]
                # count uncovered and not-yet-covered among those
                # uncovered[idxs] and ~partial_covered[idxs]
                count = int(np.count_nonzero(uncovered[idxs] & ~partial_covered[idxs]))
                if count > best_gain:
                    best_gain = count
                    best_x = x
            if best_x is None:
                best_x = candidates[0]

            block.append(best_x)
            in_block_mask |= (1 << best_x)

            # update partial_covered: add t-subsets that contain best_x and are now
            # fully contained in block. Only t-subsets containing best_x can newly be covered.
            for sidx in point_subsets[best_x]:
                m = masks[sidx]
                # check all members in block
                if (m & in_block_mask) == m:
                    partial_covered[sidx] = True

        block.sort()
        blocks.append(block)

        # remove from uncovered every t-subset contained in this full block
        for c in itertools.combinations(block, t):
            m = sum(1 << x for x in c)
            idx = mask_to_idx.get(m)
            if idx is not None:
                uncovered[idx] = False

        if time.time() > deadline:
            break

    return blocks
# EVOLVE-BLOCK-END
