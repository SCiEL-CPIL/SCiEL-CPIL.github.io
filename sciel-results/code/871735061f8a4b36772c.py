"""Covering design C(v,k,t): minimize number of k-blocks covering all t-subsets.

Entry point: construct(v, k, t) -> list of blocks (each a list of k distinct ints in 0..v-1).
Fast greedy cover + local-search reduction (redundancy removal + add/remove moves),
all under an explicit wall-clock deadline.
"""
import time
import random
import itertools
import numpy as np

# Wall-clock budget (evaluator kills at 150s; keep a safety margin).
TIME_LIMIT = 115.0


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _index_map(v, t):
    m = {}
    for i, c in enumerate(itertools.combinations(range(v), t)):
        m[c] = i
    return m


def _greedy(v, k, t, tsubs, imap, deadline, rng):
    n = len(tsubs)
    W = max(1, (n + 63) // 64)
    uncovered = np.ones(W, dtype=np.uint64)
    if n % 64:
        uncovered[-1] &= (1 << (n % 64)) - 1

    blocks = []
    used = [False] * v

    while True:
        if time.time() > deadline:
            break
        nz = np.nonzero(uncovered)[0]
        if nz.size == 0:
            break
        w0 = int(nz[0])
        base = w0 * 64
        mask = uncovered[w0]
        # lowest set bit -> first uncovered t-subset
        lb = (mask & (-mask)).bit_length() - 1
        start = tsubs[base + lb]

        block = [start[0]]
        used[start[0]] = True

        # Precompute, for each point, the bitmask of t-subsets that contain it.
        pt_mask = {}
        for x in range(v):
            m = 0
            for rest in itertools.combinations(range(v), t - 1):
                idx = imap[tuple(sorted(rest + (x,)))]
                m |= 1 << (idx % 64)
                if idx >= 64:
                    pass
            pt_mask[x] = m

        # For the current block, track the bitmask of t-subsets it already covers.
        block_mask = np.zeros(W, dtype=np.uint64)
        for c in itertools.combinations(block, t):
            idx = imap[c]
            block_mask[idx // 64] |= 1 << (idx % 64)

        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if used[x]:
                    continue
                # new t-subsets added by x = those containing x, in block_mask, uncovered
                # = pt_mask[x] & block_mask & uncovered
                m = pt_mask[x]
                newcount = 0
                for w in range(W):
                    newcount += int((m[w] & block_mask[w] & uncovered[w]).bit_count())
                if newcount > best_gain:
                    best_gain = newcount
                    best_x = x
            if best_x < 0:
                # no point adds coverage; fill with any unused point
                for x in range(v):
                    if not used[x]:
                        best_x = x
                        break
            block.append(best_x)
            used[best_x] = True
            # update block_mask with new t-subsets
            for rest in itertools.combinations(block[:-1], t - 1):
                idx = imap[tuple(sorted(rest + (best_x,)))]
                block_mask[idx // 64] |= 1 << (idx % 64)

        block.sort()
        blocks.append(block)
        # mark covered
        for c in itertools.combinations(block, t):
            idx = imap[c]
            uncovered[idx // 64] &= ~(1 << (idx % 64))

    return blocks


def _local_search(v, k, t, blocks, tsubs, imap, deadline, rng):
    n = len(tsubs)
    W = max(1, (n + 63) // 64)
    cover = np.zeros(W, dtype=np.uint64)
    for b in blocks:
        for c in itertools.combinations(b, t):
            idx = imap[c]
            cover[idx // 64] |= 1 << (idx % 64)

    # ensure full coverage
    def is_full(cov):
        for w in range(W):
            if cov[w] != 0xFFFFFFFFFFFFFFFF:
                return False
        # mask the tail
        if n % 64:
            if cov[-1] != (1 << (n % 64)) - 1:
                return False
        return True

    if not is_full(cover):
        return blocks

    def remove_block(cov, b):
        for c in itertools.combinations(b, t):
            idx = imap[c]
            cov[idx // 64] &= ~(1 << (idx % 64))

    def add_block(cov, b):
        for c in itertools.combinations(b, t):
            idx = imap[c]
            cov[idx // 64] |= 1 << (idx % 64)

    # --- Pass 1: remove redundant blocks ---
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(blocks):
            b = blocks[i]
            tmp = cover.copy()
            remove_block(tmp, b)
            if is_full(tmp):
                cover = tmp
                blocks.pop(i)
                changed = True
            else:
                i += 1

    # --- Pass 2: add-random / remove-redundant moves ---
    all_blocks_set = set(tuple(b) for b in blocks)
    while time.time() < deadline:
        # try to add a random block that enables removals
        cand = tuple(sorted(rng.sample(range(v), k)))
        if cand in all_blocks_set:
            continue
        tmp = cover.copy()
        add_block(tmp, cand)
        blocks.append(list(cand))
        all_blocks_set.add(cand)
        # now greedily remove redundant blocks
        removed_any = False
        j = 0
        while j < len(blocks):
            bb = blocks[j]
            t2 = tmp.copy()
            remove_block(t2, bb)
            if is_full(t2):
                tmp = t2
                all_blocks_set.discard(tuple(bb))
                blocks.pop(j)
                removed_any = True
            else:
                j += 1
        if removed_any:
            cover = tmp
        else:
            # revert the added block
            all_blocks_set.discard(cand)
            blocks.pop()
            # occasionally break out if stuck
            if len(blocks) < 200:
                break

    return blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    rng = random.Random(12345)

    tsubs = _all_tsubsets(v, t)
    imap = _index_map(v, t)

    blocks = _greedy(v, k, t, tsubs, imap, deadline, rng)

    if time.time() < deadline - 1.0:
        blocks = _local_search(v, k, t, blocks, tsubs, imap, deadline, rng)

    return blocks
