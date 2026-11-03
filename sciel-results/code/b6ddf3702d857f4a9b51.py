"""Global set-cover greedy for C(19,8,5) with incremental gain updates.

Strategy:
  - Precompute, for every 8-block, the list of 5-subset indices it covers, and
    for every 5-subset, the list of blocks that cover it.
  - Maintain a lazy-deletion max-heap keyed by "number of currently-uncovered
    5-subsets this block covers".  After adding a block, only the gains of
    blocks that share a 5-subset with it can change, so we lazily recompute
    their gains (stale heap entries are discarded).
  - Randomized restarts with different first-block seeds; keep the best
    (fewest-block) complete covering found before the deadline.
  - A local-improvement pass drops redundant blocks.
"""
import time
import random
import heapq
import itertools
import numpy as np


def _block_cover_indices(block, t, block_to_t5):
    out = []
    for S in itertools.combinations(block, t):
        sm = 0
        for y in S:
            sm |= 1 << y
        idx = block_to_t5.get(sm)
        if idx is not None:
            out.append(idx)
    return out


def construct(v, k, t):
    deadline = time.time() + 112.0  # safety margin before 120s kill
    rng = random.Random(12345)

    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)

    # map t-subset bitmask -> index
    block_to_t5 = {}
    for i, c in enumerate(all_t):
        m = 0
        for x in c:
            m |= 1 << x
        block_to_t5[m] = i

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # coverage lists
    block_cover = [_block_cover_indices(blk, t, block_to_t5)
                   for blk in all_blocks]

    # blocks_sharing[idx] = list of block indices that cover t-subset idx
    blocks_sharing = [[] for _ in range(n_t)]
    for b, cov in enumerate(block_cover):
        for idx in cov:
            blocks_sharing[idx].append(b)

    def greedy_pass(seed_block):
        """One global greedy pass. Returns (list_of_block_indices, uncovered)."""
        covered = np.zeros(n_t, dtype=bool)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.array([int(np.count_nonzero(uncovered[bc]))
                         for bc in block_cover], dtype=np.int32)
        version = np.zeros(n_blocks, dtype=np.int32)
        heap = [(-int(gain[b]), b, 0) for b in range(n_blocks)]
        heapq.heapify(heap)

        blocks = []
        seen = set()

        # optional seed
        if seed_block is not None:
            blocks.append(seed_block)
            seen.add(seed_block)
            for idx in block_cover[seed_block]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    covered[idx] = True
            for idx in block_cover[seed_block]:
                for b2 in blocks_sharing[idx]:
                    if b2 != seed_block:
                        version[b2] += 1

        unc = n_t - int(np.count_nonzero(covered))

        while unc > 0:
            if time.time() > deadline:
                break
            b = None
            while heap:
                neg_g, cand, ver = heapq.heappop(heap)
                if ver != version[cand]:
                    continue
                g = -neg_g
                if g <= 0:
                    return blocks, unc
                b = cand
                break
            if b is None:
                return blocks, unc

            blocks.append(b)
            seen.add(b)
            for idx in block_cover[b]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    covered[idx] = True
                    unc -= 1
            for idx in block_cover[b]:
                for b2 in blocks_sharing[idx]:
                    if b2 != b:
                        version[b2] += 1

        return blocks, unc

    best_blocks = None
    best_unc = None

    # deterministic first pass
    res, unc = greedy_pass(None)
    if unc == 0:
        best_blocks, best_unc = res, 0
    elif best_unc is None or unc < best_unc:
        best_blocks, best_unc = res, unc

    # randomized restarts
    while time.time() < deadline - 1.0:
        seed = rng.randrange(n_blocks)
        res, unc = greedy_pass(seed)
        if unc == 0:
            if best_blocks is None or len(res) < len(best_blocks):
                best_blocks, best_unc = res, 0
        elif best_unc is None or unc < best_unc:
            best_blocks, best_unc = res, unc

    # local improvement: drop redundant blocks from the best complete covering
    if best_unc == 0 and best_blocks is not None and time.time() < deadline - 2.0:
        blocks = list(best_blocks)
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            cnt = np.zeros(n_t, dtype=np.int32)
            for b in blocks:
                for idx in block_cover[b]:
                    cnt[idx] += 1
            new_blocks = []
            for b in blocks:
                redundant = True
                for idx in block_cover[b]:
                    if cnt[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        best_blocks = blocks

    # if still incomplete, guarantee validity with a plain greedy top-up
    if best_blocks is None or best_unc > 0:
        covered = np.zeros(n_t, dtype=bool)
        blocks = []
        unc = n_t
        while unc > 0 and time.time() < deadline:
            best_b = None
            best_g = -1
            for b in range(n_blocks):
                g = int(np.count_nonzero(~covered[block_cover[b]]))
                if g > best_g:
                    best_g = g
                    best_b = b
            if best_b is None or best_g <= 0:
                break
            blocks.append(best_b)
            for idx in block_cover[best_b]:
                if not covered[idx]:
                    covered[idx] = True
                    unc -= 1
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks

    # final safety: ensure every t-subset is covered
    if best_blocks is None:
        best_blocks = []
    covered = np.zeros(n_t, dtype=bool)
    for b in best_blocks:
        for idx in block_cover[b]:
            covered[idx] = True
    if not covered.all() and time.time() < deadline:
        while not covered.all() and time.time() < deadline:
            best_b = None
            best_g = -1
            for b in range(n_blocks):
                g = int(np.count_nonzero(~covered[block_cover[b]]))
                if g > best_g:
                    best_g = g
                    best_b = b
            if best_b is None or best_g <= 0:
                break
            best_blocks.append(best_b)
            for idx in block_cover[best_b]:
                covered[idx] = True

    return [list(all_blocks[b]) for b in best_blocks]
