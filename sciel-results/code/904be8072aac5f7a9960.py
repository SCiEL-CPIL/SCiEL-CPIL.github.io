# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}

    all_t1subs = list(itertools.combinations(range(v), t - 1))
    n_t1subs = len(all_t1subs)
    t1sub_to_idx = {frozenset(c): i for i, c in enumerate(all_t1subs)}

    # tsub_for[x, s_idx] = index of {x}+S_s in all_tsubs, or -1 if x in S_s
    tsub_for = np.full((v, n_t1subs), -1, dtype=np.int32)
    for s_idx, s in enumerate(all_t1subs):
        s_set = set(s)
        for x in range(v):
            if x not in s_set:
                full = tuple(sorted(s + (x,)))
                tsub_for[x, s_idx] = tsub_to_idx[frozenset(full)]

    def _greedy(rng, dl):
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while uncovered.any():
            if time.time() > dl:
                break

            uncovered_indices = np.where(uncovered)[0]
            seed_idx = rng.choice(uncovered_indices)
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)

            def _t1_idxs(blk):
                return [t1sub_to_idx[frozenset(c)] for c in itertools.combinations(blk, t - 1)]

            t1_idxs = _t1_idxs(block)

            while len(block) < k:
                if time.time() > dl:
                    break

                # Vectorized gain: for each x not in block, count uncovered tsub_for[x, t1_idxs]
                gains = np.zeros(v, dtype=np.int64)
                for x in range(v):
                    if x in block_set:
                        continue
                    idxs = tsub_for[x, t1_idxs]
                    valid = idxs >= 0
                    if valid.any():
                        gains[x] = uncovered[idxs[valid]].sum()

                mask = np.ones(v, dtype=bool)
                for x in block:
                    mask[x] = False
                valid_pts = mask & (gains > 0)

                if not valid_pts.any():
                    candidates = np.where(mask)[0]
                    if len(candidates) == 0:
                        break
                    best_point = int(rng.choice(candidates))
                else:
                    max_gain = np.max(gains[valid_pts])
                    candidates = np.where(valid_pts & (gains == max_gain))[0]
                    best_point = int(rng.choice(candidates))

                block.append(best_point)
                block_set.add(best_point)
                t1_idxs = _t1_idxs(block)

            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[frozenset(c)]
                    uncovered[idx] = False

        return [list(b) for b in blocks]

    def _local_search(blocks, dl):
        blocks = [list(b) for b in blocks]
        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[frozenset(c)]] = True
            block_masks.append(mask)
            cov_count += mask

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > dl:
                    return blocks
                mask = block_masks[i]
                if not np.any(cov_count[mask] <= 1):
                    cov_count -= mask
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        return blocks

    best_blocks = None
    best_len = float('inf')

    rng0 = random.Random(42)
    baseline = _greedy(rng0, deadline)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)

    restart = 0
    while time.time() < deadline - 30:
        restart += 1
        rng = random.Random(restart * 7919 + 13)
        blocks = _greedy(rng, deadline)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    if best_blocks and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, deadline)

    return best_blocks if best_blocks else []

# EVOLVE-BLOCK-END
