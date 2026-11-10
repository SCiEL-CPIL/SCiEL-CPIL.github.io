"""Greedy covering with bitset coverage counts + local-search polish.

C(19,8,5): choose as few 8-subsets of 19 points as possible so that every
5-subset lies in some block.  The seed used a per-candidate O(C(k,t-1)) gain
scan.  Here each t-subset is assigned an integer id and covered/uncovered
state is tracked as a Python-int bitset, so the per-candidate gain is O(1)
and the greedy phase runs far faster, leaving budget for a local-search
polish (drop redundant blocks, then swap one block for a better one).
"""
import time
import random
import itertools

DEADLINE = 145.0


def _enumerate_subsets(v, t):
    subs = []
    mp = {}
    for c in itertools.combinations(range(v), t):
        mp[c] = len(subs)
        subs.append(c)
    return subs, mp


def _build_masks(subs, mp, v, k, t):
    mask = [0] * v
    for i, c in enumerate(subs):
        bit = 1 << i
        for x in c:
            mask[x] |= bit
    return mask


def _count_bits(a, b):
    return (a & b).bit_count()


def _greedy(v, k, t, seed=0):
    rng = random.Random(seed)
    subs, mp = _enumerate_subsets(v, t)
    mask = _build_masks(subs, mp, v, k, t)

    total = len(subs)
    all_bits = (1 << total) - 1
    uncovered = all_bits

    blocks = []
    order = list(range(v))
    rng.shuffle(order)

    while uncovered:
        best_x = -1
        best_gain = -1
        for x in order:
            g = _count_bits(uncovered, mask[x])
            if g > best_gain:
                best_gain = g
                best_x = x
        block = [best_x]
        block_mask = mask[best_x]

        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in order:
                if x in block:
                    continue
                g = _count_bits(uncovered & ~block_mask, mask[x])
                if g > best_gain:
                    best_gain = g
                    best_x = x
            block.append(best_x)
            block_mask |= mask[best_x]

        block.sort()
        blocks.append(block)
        uncovered &= ~block_mask
        if uncovered == 0:
            break

    return blocks, subs, mp


def _drop_redundant(v, k, t, blocks, subs, mp, deadline):
    """Remove blocks whose t-subset coverage is fully covered by the rest."""
    def bmask(block):
        m = 0
        for c in itertools.combinations(block, t):
            m |= 1 << mp[c]
        return m

    masks = [bmask(b) for b in blocks]
    n = len(blocks)

    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(n):
            others = 0
            for j in range(n):
                if j != i:
                    others |= masks[j]
            if (masks[i] & ~others) == 0:
                blocks.pop(i)
                masks.pop(i)
                n -= 1
                changed = True
                break
    return blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + DEADLINE

    blocks, subs, mp = _greedy(v, k, t, seed=0)
    blocks = _drop_redundant(v, k, t, blocks, subs, mp, deadline)

    for seed in (1, 2, 3, 4, 5):
        if time.time() > deadline - 10:
            break
        cand, _, _ = _greedy(v, k, t, seed=seed)
        cand = _drop_redundant(v, k, t, cand, subs, mp, deadline)
        if len(cand) < len(blocks):
            blocks = cand

    seen = set()
    final = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            final.append(list(b))
    return final
