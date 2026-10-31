"""Fresh design: bitmask-accelerated greedy covering with a hard deadline + repair."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin well under 120s
    rng = random.Random(12345)

    # Precompute, for every k-subset mask, the list of t-subset masks it covers.
    masks_by_mask = {}
    for comb in itertools.combinations(range(v), k):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(sm)
        masks_by_mask[mask] = subs

    # Enumerate all t-subset masks in a fixed order (used for seeding/repair).
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        sm = 0
        for p in sub:
            sm |= (1 << p)
        tsub_masks.append(sm)

    # cover_count[i] = how many blocks cover tsub_masks[i]; uncovered = count==0
    cover_count = np.zeros(len(tsub_masks), dtype=np.int32)
    uncovered = cover_count == 0

    # map tsub-mask -> index for quick lookups
    idx_of = {m: i for i, m in enumerate(tsub_masks)}

    blocks = []
    blocks_set = set()
    n = v

    def deadline_hit():
        return time.time() > deadline

    def add_block(mask):
        blocks.append(mask)
        blocks_set.add(mask)
        for sm in masks_by_mask[mask]:
            cover_count[idx_of[sm]] += 1
        uncovered[:] = cover_count == 0
        return uncovered.sum()

    # Greedy: repeatedly open a block on the first uncovered t-subset, then grow it.
    while uncovered.any() and not deadline_hit():
        # seed: smallest uncovered t-subset (deterministic)
        first = int(np.argmax(uncovered))
        seed_mask = tsub_masks[first]
        block_mask = seed_mask
        # points already in block
        in_block = [p for p in range(n) if (block_mask >> p) & 1]
        # covered-so-far t-subset masks (set) for marginal gain
        covered = set(masks_by_mask.get(block_mask, [seed_mask]))
        # ensure seed covers its own t-subsets
        for sm in masks_by_mask.get(block_mask, []):
            covered.add(sm)

        while len(in_block) < k and not deadline_hit():
            best_p = None
            best_gain = -1
            best_div = -1
            for p in range(n):
                if (block_mask >> p) & 1:
                    continue
                # marginal gain: new t-subsets formed by adding p
                gain = 0
                # t-subsets containing p and (t-1) from current block
                for rest in itertools.combinations(in_block, t - 1):
                    sm = 0
                    for q in rest:
                        sm |= (1 << q)
                    sm |= (1 << p)
                    if cover_count[idx_of[sm]] == 0:
                        gain += 1
                if gain < best_gain:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_div = gain
                else:
                    # tie-break: prefer point with fewer existing blocks containing it (diversity)
                    div = 0
                    for b in blocks:
                        if (b >> p) & 1:
                            div += 1
                    if div < best_div:
                        best_div = div
                        best_p = p
            if best_p is None or best_gain <= 0:
                # no beneficial point; fill randomly to reach k (keeps progress)
                candidates = [p for p in range(n) if not ((block_mask >> p) & 1)]
                if not candidates:
                    break
                if len(candidates) > (k - len(in_block)):
                    candidates = rng.sample(candidates, k - len(in_block))
                for p in candidates:
                    block_mask |= (1 << p)
                    in_block.append(p)
                break
            block_mask |= (1 << best_p)
            in_block.append(best_p)
            # update covered set (for potential future use)
            for rest in itertools.combinations(in_block[:-1], t - 1):
                sm = 0
                for q in rest:
                    sm |= (1 << q)
                sm |= (1 << best_p)
                covered.add(sm)

        if block_mask not in blocks_set:
            add_block(block_mask)

    # REPAIR: guarantee validity even if greedy stopped early.
    # For each uncovered t-subset, open a block covering it (deterministic, fast).
    uncovered_idx = np.where(cover_count == 0)[0]
    for i in uncovered_idx:
        if deadline_hit():
            # even if we run out, we MUST cover; do a minimal fill with no time check
            pass
        sm = tsub_masks[i]
        # build a k-subset containing this t-subset
        have = [p for p in range(n) if (sm >> p) & 1]
        need = k - len(have)
        extra = []
        for p in range(n):
            if need <= 0:
                break
            if not ((sm >> p) & 1):
                extra.append(p)
                need -= 1
        block = have + extra
        block.sort()
        mask = 0
        for p in block:
            mask |= (1 << p)
        if mask not in blocks_set:
            add_block(mask)

    # Convert masks back to sorted lists of points.
    result = []
    for mask in blocks:
        pts = [p for p in range(n) if (mask >> p) & 1]
        pts.sort()
        result.append(pts)
    return result
# EVOLVE-BLOCK-END
