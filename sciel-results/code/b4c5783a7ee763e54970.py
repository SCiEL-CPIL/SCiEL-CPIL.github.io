# EVOLVE-BLOCK-START
import itertools
import time
import random


def _build_tables(v, t):
    """Precompute all t-subsets as bit values and pair-index gain tables."""
    all_tsubs = list(itertools.combinations(range(v), t))
    idx = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # pair_index[R][x] = bit value of t-subset R U {x} (or 0 if x in R)
    pair_index = {}
    for R in itertools.combinations(range(v), t - 1):
        bits = {}
        Rset = set(R)
        for x in range(v):
            if x in Rset:
                bits[x] = 0
            else:
                bits[x] = 1 << idx[tuple(sorted(R + (x,)))]
        pair_index[R] = bits

    # pair_list[R] = list of (x, bit) for x not in R (for fast gain accumulation)
    pair_list = {}
    for R, bits in pair_index.items():
        lst = [(x, b) for x, b in bits.items() if b]
        pair_list[R] = lst

    return all_tsubs, idx, n_tsubs, pair_index, pair_list


def _greedy_cover(v, k, t, pair_index, pair_list, idx, all_tsubs,
                  start_points, deadline, rng):
    """Run one greedy covering run starting from the given seed points.

    Returns (blocks, covered_mask) or None if it cannot finish in time.
    """
    full_mask = (1 << len(all_tsubs)) - 1
    covered = 0

    # seed the first block with start_points if provided, else empty
    block = []
    if start_points:
        for p in start_points:
            if p not in block:
                block.append(p)

    blocks = []

    while True:
        if time.time() > deadline:
            return None

        # finish current block
        while len(block) < k:
            if time.time() > deadline:
                return None
            block_set = set(block)
            # accumulate gains for each candidate point
            gains = {}
            for R in itertools.combinations(block, t - 1):
                for x, bit in pair_list[R]:
                    if x in block_set:
                        continue
                    if covered & bit:
                        continue
                    gains[x] = gains.get(x, 0) + 1
            if not gains:
                # no uncovered t-subset can be extended with current block prefix
                # pick any unused point
                used = set(block)
                cands = [x for x in range(v) if x not in used]
                if not cands:
                    break
                block.append(rng.choice(cands))
                continue
            best = max(gains.items(), key=lambda kv: (kv[1], -kv[0]))
            block.append(best[0])

        block.sort()
        blocks.append(block)

        # update covered mask with all t-subsets of this block
        for sub in itertools.combinations(block, t):
            covered |= 1 << idx[sub]

        if covered == full_mask:
            return blocks, covered

        # start next block: pick the first (lowest) uncovered t-subset as seed
        # find an uncovered t-subset to seed the next block
        # iterate over all_tsubs to find one not covered
        seed = None
        for sub in all_tsubs:
            if not (covered & (1 << idx[sub])):
                seed = list(sub)
                break
        if seed is None:
            return blocks, covered
        # seed next block with a few points from an uncovered subset
        block = seed[:min(t, k)]

    return blocks, covered


def construct(v, k, t):
    deadline = time.time() + 115.0   # safety margin below the 120s limit

    all_tsubs, idx, n_tsubs, pair_index, pair_list = _build_tables(v, t)
    full_mask = (1 << n_tsubs) - 1

    rng = random.Random(12345)

    # deterministic baseline first (seeded by the smallest uncovered subset)
    best_blocks = None
    best_len = None

    # --- deterministic run (no random seed, start empty) ---
    res = _greedy_cover(v, k, t, pair_index, pair_list, idx, all_tsubs,
                        None, deadline, rng)
    if res is not None:
        blocks, covered = res
        if covered == full_mask and (best_len is None or len(blocks) < best_len):
            best_blocks = blocks
            best_len = len(blocks)

    # --- randomized restarts with diverse strategies ---
    # Strategy A: random start points
    # Strategy B: rarity-based (start from a random uncovered t-subset)
    while time.time() < deadline:
        strategy = rng.random()
        if strategy < 0.5:
            # random start: pick 1-3 random points
            n_seed = rng.randint(1, min(3, k))
            start = rng.sample(range(v), n_seed)
        else:
            # rarity-based: pick a random t-subset as the seed (forces coverage of it)
            start = list(rng.choice(all_tsubs))
            if len(start) > k:
                start = start[:k]

        res = _greedy_cover(v, k, t, pair_index, pair_list, idx, all_tsubs,
                            start, deadline, rng)
        if res is not None:
            blocks, covered = res
            if covered == full_mask and (best_len is None or len(blocks) < best_len):
                best_blocks = blocks
                best_len = len(blocks)

    if best_blocks is None:
        # should not happen; fall back to a minimal valid construction
        # build a trivial covering: every t-subset extended to a k-block
        # (may exceed limits but is valid)
        blocks = []
        seen = set()
        for sub in all_tsubs:
            block = list(sub)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
            block.sort()
            key = tuple(block)
            if key not in seen:
                seen.add(key)
                blocks.append(block)
        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
