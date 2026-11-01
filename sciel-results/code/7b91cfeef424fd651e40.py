"""Covering design C(v,k,t) - bitmask-accelerated greedy with randomized restarts and a deadline guard."""
import itertools
import time
import random


def construct(v, k, t):
    t0 = time.time()
    DEADLINE = t0 + 145.0   # safety margin below the 150 s wall-clock limit

    # Precompute masks over all t-subsets.
    combs = list(itertools.combinations(range(v), t))
    M = len(combs)
    idx = {c: i for i, c in enumerate(combs)}

    point_mask = [0] * v
    for i, c in enumerate(combs):
        for p in c:
            point_mask[p] |= (1 << i)

    pair_mask = {}
    for i, c in enumerate(combs):
        for a in range(t):
            for b in range(a + 1, t):
                key = (c[a], c[b])
                pair_mask[key] = pair_mask.get(key, 0) | (1 << i)

    full_mask = (1 << M) - 1
    all_points = list(range(v))

    def popcount(x):
        return bin(x).count('1')

    def greedy(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        blocks = []
        while uncovered:
            # Pick the first (deterministic) uncovered t-subset to anchor the block.
            anchor = None
            for i in range(M):
                if (uncovered >> i) & 1:
                    anchor = combs[i]
                    break
            if anchor is None:
                break

            # Build the block: seed with the anchor points, then greedily add the
            # candidate that covers the most still-uncovered t-subsets.
            if seed != 0:
                # Shuffle which anchor points to place first for variety.
                shuffled = list(anchor)
                rng.shuffle(shuffled)
                block = shuffled[:1]
                for p in shuffled[1:]:
                    block.append(p)
            else:
                block = list(anchor)

            while len(block) < k:
                if len(block) >= t:
                    # Exact gain: t-subsets that contain a candidate and an existing
                    # point, and are still uncovered.
                    prev = block[-1]
                    pm = point_mask[prev]
                    best = -1
                    best_gain = -1
                    for cand in all_points:
                        if cand in block:
                            continue
                        g = popcount(point_mask[cand] & pm & uncovered)
                        if g > best_gain:
                            best_gain = g
                            best = cand
                    if best < 0:
                        break
                    block.append(best)
                else:
                    # Not enough points yet for an exact gain; complete with unused points.
                    for cand in all_points:
                        if cand not in block:
                            block.append(cand)
                            if len(block) == k:
                                break

            # Remove all t-subsets covered by this block.
            cover = 0
            for c in itertools.combinations(block, t):
                cover |= (1 << idx[c])
            uncovered &= ~cover
            blocks.append(block)
        return blocks

    # Phase 1: deterministic greedy (seed 0) gives a guaranteed valid result.
    best = greedy(0)
    best_len = len(best)

    # Phase 2: randomized restarts to try to beat the greedy count.
    seed = 1
    while time.time() < DEADLINE:
        cur = greedy(seed)
        if len(cur) < best_len:
            best = cur
            best_len = len(cur)
        seed += 1

    return best
