# EVOLVE-BLOCK-START
import itertools
import random
import time


def _all_tsubs(v, t):
    return list(itertools.combinations(range(v), t))


def _block_covers(block, t):
    return [set(c) for c in itertools.combinations(block, t)]


def _make_cov(v, k, t, tsubs):
    # cov[point] = bitset of t-subsets containing that point
    cov = [0] * v
    for i, s in enumerate(tsubs):
        mask = 1 << i
        for p in s:
            cov[p] |= mask
    return cov


def _greedy_construct(v, k, t, rng, deadline, cov, tsubs, n_tsubs):
    all_mask = (1 << n_tsubs) - 1
    uncovered = all_mask
    blocks = []
    while uncovered:
        # pick a seed t-subset (an uncovered one)
        idx = (uncovered & -uncovered).bit_length() - 1
        seed = tsubs[idx]
        best_cov = 0
        best_block = None
        best_covered = 0
        for _ in range(4):  # candidate blocks
            if best_block is not None:
                if rng.random() < 0.5:
                    block = list(seed)
                else:
                    block = list(seed)
                    while len(block) < k:
                        best = -1
                        bestg = -1
                        for x in range(v):
                            if x not in block:
                                g = (cov[x] & uncovered).bit_count()
                                if g > bestg:
                                    bestg = g
                                    best = x
                        block.append(best)
            else:
                block = list(seed)
                while len(block) < k:
                    best = -1
                    bestg = -1
                    for x in range(v):
                        if x not in block:
                            g = (cov[x] & uncovered).bit_count()
                            if g > bestg:
                                bestg = g
                                best = x
                    block.append(best)
            block.sort()
            covered = 0
            for s in _block_covers(block, t):
                covered |= cov[s[0]]
                for p in s[1:]:
                    covered |= cov[p]
            if covered & ~uncovered:  # covers at least one new
                if covered.bit_count() > best_covered:
                    best_covered = covered.bit_count()
                    best_block = block
                    best_cov = covered
        if best_block is None:
            # fallback: just the seed block
            block = list(seed)
            while len(block) < k:
                block.append(rng.randrange(v))
            block = sorted(set(block))
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            best_block = block
            best_cov = 0
            for s in _block_covers(block, t):
                best_cov |= cov[s[0]]
                for p in s[1:]:
                    best_cov |= cov[p]
        blocks.append(best_block)
        uncovered &= ~best_cov
        if time.time() > deadline:
            break
    return blocks, uncovered


def _local_search(v, k, t, blocks, cov, tsubs, n_tsubs, rng, deadline):
    all_mask = (1 << n_tsubs) - 1
    # compute coverage per block
    def block_mask(block):
        m = 0
        for s in _block_covers(block, t):
            m |= cov[s[0]]
            for p in s[1:]:
                m |= cov[p]
        return m
    masks = [block_mask(b) for b in blocks]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # drop redundant blocks
        new_blocks = []
        new_masks = []
        for i, b in enumerate(blocks):
            if i < len(new_masks) and (masks[i] & ~(new_masks[i] | all_mask)) == 0:
                # covered by previous? check masks[i] subset of union of new_masks
                union = 0
                for nm in new_masks:
                    union |= nm
                if (masks[i] & ~union) == 0:
                    improved = True
                    continue
            new_blocks.append(b)
            new_masks.append(masks[i])
        blocks = new_blocks
        masks = new_masks
        # swap points to cover more
        covered = 0
        for m in masks:
            covered |= m
        uncovered = all_mask & ~covered
        for i, b in enumerate(blocks):
            if time.time() > deadline:
                break
            bset = set(b)
            # value of each point in block = cov[p] & (uncovered | (masks[i] & ~covered_without_i))
            # simpler: try removing point j, see uncovered delta
            for j in range(k):
                pj = b[j]
                # mask of block without pj
                m_without = 0
                for s in _block_covers([x for x in b if x != pj], t):
                    m_without |= cov[s[0]]
                    for p in s[1:]:
                        m_without |= cov[p]
                # points only covered by pj within this block
                only = masks[i] & ~m_without
                # try replacing pj with x
                best = -1
                bestgain = 0
                for x in range(v):
                    if x in bset:
                        continue
                    m_new = m_without | cov[x]
                    gain = (m_new & uncovered).bit_count() - (m_without & uncovered).bit_count()
                    # also count new coverage of 'only' not elsewhere
                    if gain > bestgain:
                        bestgain = gain
                        best = x
                if best >= 0 and bestgain > 0:
                    b[j] = best
                    bset.discard(pj)
                    bset.add(best)
                    b.sort()
                    masks[i] = m_without | cov[best]
                    improved = True
                    covered |= masks[i]
                    uncovered = all_mask & ~covered
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill
    tsubs = _all_tsubs(v, t)
    n_tsubs = len(tsubs)
    cov = _make_cov(v, k, t, tsubs)
    rng = random.Random(12345)
    blocks, uncovered = _greedy_construct(v, k, t, rng, deadline, cov, tsubs, n_tsubs)
    if uncovered == 0 and time.time() < deadline - 2.0:
        blocks = _local_search(v, k, t, blocks, cov, tsubs, n_tsubs, rng, deadline)
    # ensure validity: if not fully covered, keep adding greedy blocks until covered or deadline
    def block_mask(block):
        m = 0
        for s in _block_covers(block, t):
            m |= cov[s[0]]
            for p in s[1:]:
                m |= cov[p]
        return m
    covered = 0
    for b in blocks:
        covered |= block_mask(b)
    uncovered = ((1 << n_tsubs) - 1) & ~covered
    while uncovered and time.time() < deadline:
        idx = (uncovered & -uncovered).bit_length() - 1
        seed = tsubs[idx]
        block = list(seed)
        while len(block) < k:
            best = -1
            bestg = -1
            for x in range(v):
                if x not in block:
                    g = (cov[x] & uncovered).bit_count()
                    if g > bestg:
                        bestg = g
                        best = x
            block.append(best)
        block.sort()
        blocks.append(block)
        covered |= block_mask(block)
        uncovered = ((1 << n_tsubs) - 1) & ~covered
    return blocks
# EVOLVE-BLOCK-END
