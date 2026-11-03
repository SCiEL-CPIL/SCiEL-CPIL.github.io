"""Improved covering design solver with randomized greedy, randomized restarts, and redundancy removal.

Algorithm family: randomized greedy set-cover with bitmask acceleration.
- Precompute all t-subsets and per-point gain data using Python big-int bitmasks.
- Greedy step: at each position of a k-block, pick the point maximizing the number of
  newly-coverable t-subsets (those whose (S \ {x}) is already a subset of the partial block).
- Randomized: multiple seeds, random tie-breaking, random initial seed t-subset per block.
- Post-processing: remove blocks that are fully redundant (their t-subsets are all covered by others).
- Keep the best result found before the deadline.
"""
import time
import random
import itertools


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)

    # Bitmask representation of each t-subset
    tsub_mask = [0] * num_tsubsets
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsub_mask[i] = mask

    # For each point x, precompute list of (tsub_idx, req_mask) where req_mask = tsub_mask[i] & ~(1<<x)
    # Gain of adding x to a partial block B: count of uncovered i where x in tsub i and req_mask subset of B
    point_data = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        m = tsub_mask[i]
        for p in ts:
            point_data[p].append((i, m & ~(1 << p)))

    def greedy_cover(rng_seed):
        rng = random.Random(rng_seed)
        uncovered = bytearray([1]) * num_tsubsets
        blocks = []
        remaining = num_tsubsets

        while remaining > 0:
            if time.time() > deadline:
                return None

            # Randomly pick an uncovered t-subset to seed this block
            idx = rng.randrange(num_tsubsets)
            while not uncovered[idx]:
                idx = (idx + 1) % num_tsubsets

            seed_tsub = list(all_tsubsets[idx])
            rng.shuffle(seed_tsub)
            block = seed_tsub[:]
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)

            # Greedily add points
            while len(block) < k:
                if time.time() > deadline:
                    return None

                best_gain = -1
                best_point = -1
                for x in range(v):
                    if (block_mask >> x) & 1:
                        continue
                    gain = 0
                    for ts_idx, req_mask in point_data[x]:
                        if not uncovered[ts_idx]:
                            continue
                        if (req_mask & ~block_mask) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point != -1:
                        if rng.random() < 0.5:
                            best_point = x

                if best_point == -1 or best_gain <= 0:
                    candidates = [x for x in range(v) if not ((block_mask >> x) & 1)]
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break

                block.append(best_point)
                block_mask |= (1 << best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts_idx in range(num_tsubsets):
                if not uncovered[ts_idx]:
                    continue
                if (tsub_mask[ts_idx] & ~block_mask) == 0:
                    uncovered[ts_idx] = 0
                    remaining -= 1

        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        n = len(blocks)
        # For each block, find which t-subsets it covers
        block_covers = []
        for block in blocks:
            bm = 0
            for p in block:
                bm |= (1 << p)
            covered = []
            for i in range(num_tsubsets):
                if (tsub_mask[i] & ~bm) == 0:
                    covered.append(i)
            block_covers.append(covered)

        # tsub_to_blocks: which blocks cover each t-subset
        tsub_to_blocks = [set() for _ in range(num_tsubsets)]
        for i, cov in enumerate(block_covers):
            for ts_idx in cov:
                tsub_to_blocks[ts_idx].add(i)

        new_blocks = []
        for i, block in enumerate(blocks):
            redundant = True
            for ts_idx in block_covers[i]:
                if tsub_to_blocks[ts_idx] == {i}:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(block)
        return new_blocks

    best_blocks = None
    best_count = float('inf')

    # Run greedy with multiple seeds
    seed = 0
    while time.time() < deadline - 5:
        if time.time() > deadline:
            break
        blocks = greedy_cover(seed)
        if blocks is not None:
            blocks = remove_redundant(blocks)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1

    if best_blocks is None:
        # Fallback: build one large valid covering (shouldn't happen)
        blocks = greedy_cover(0)
        if blocks is None:
            blocks = []
        best_blocks = blocks

    return best_blocks
