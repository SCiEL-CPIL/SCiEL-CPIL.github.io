"""Improved covering design C(19,8,5) with vectorized greedy + local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    uncovered_mask = np.ones(n_tsubsets, dtype=bool)
    uncovered_count = n_tsubsets
    blocks = []

    # Greedy phase with randomization for diversity
    while uncovered_count > 0:
        if time.time() > deadline:
            break

        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        block = list(all_tsubsets[start_idx])

        while len(block) < k:
            if time.time() > deadline:
                break

            block_set = set(block)
            gains = np.zeros(v, dtype=np.int32)

            # For each uncovered t-subset S, if |S \ block| == 1, increment gain for that point
            for idx in uncovered_indices:
                S = all_tsubsets[idx]
                diff = [x for x in S if x not in block_set]
                if len(diff) == 1:
                    gains[diff[0]] += 1

            for x in block:
                gains[x] = -1

            best_x = int(np.argmax(gains))
            if gains[best_x] <= 0:
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_x = random.choice(available)
                else:
                    break

            block.append(best_x)

        if len(block) < k:
            break

        block.sort()
        blocks.append(block)

        block_set = set(block)
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx.get(ts, -1)
            if idx != -1:
                if uncovered_mask[idx]:
                    uncovered_mask[idx] = False
                    uncovered_count -= 1

    # Local search: try to remove redundant blocks
    for _ in range(5):  # Multiple passes
        if time.time() > deadline:
            break

        # Compute coverage counts
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for block in blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx != -1:
                    cover_count[idx] += 1

        removed = True
        while removed and time.time() < deadline:
            removed = False
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break

                block = blocks[bi]
                block_set = set(block)

                # Find t-subsets only covered by this block
                newly_uncovered = []
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx != -1 and cover_count[idx] == 1:
                        newly_uncovered.append(idx)

                if not newly_uncovered:
                    blocks.pop(bi)
                    removed = True
                    break

                # Try to cover newly_uncovered with one new block
                union = set()
                for idx in newly_uncovered:
                    union.update(all_tsubsets[idx])

                if len(union) <= k:
                    remaining = k - len(union)
                    avail = [x for x in range(v) if x not in union]
                    if remaining > 0 and avail:
                        new_block = list(union) + random.sample(avail, remaining)
                    else:
                        new_block = list(union)
                    new_block.sort()

                    # Check if new_block is already present
                    if new_block not in blocks:
                        blocks.pop(bi)
                        blocks.append(new_block)
                        removed = True
                        break
                    else:
                        blocks.pop(bi)
                        removed = True
                        break

    return blocks
# EVOLVE-BLOCK-END
