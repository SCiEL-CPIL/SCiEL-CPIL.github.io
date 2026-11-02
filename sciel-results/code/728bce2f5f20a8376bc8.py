"""Covering design C(19,8,5) via vectorized greedy with randomized restarts,
redundancy removal, and drop-repair local search."""
import time
import random
import itertools
import numpy as np

_TIME_LIMIT = 145.0  # safety margin under the 150s kill


def _cover_mask(block, t, v):
    """Bitmask of all t-subsets contained in block (bit index = lexicographic index)."""
    m = 0
    for c in itertools.combinations(block, t):
        m |= (1 << c[0])
        for x in c[1:]:
            m |= (1 << x)
    return m


def _vectorized_greedy(uncovered, k, v, seed, rng):
    """Greedy: open block on seed (or first uncovered), add point with max gain."""
    n = len(uncovered)
    idx = np.arange(n)
    masks = np.array(uncovered, dtype=np.uint64)
    uncovered_arr = np.ones(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    blocks = []

    # seed: random uncovered subset if provided, else first uncovered
    if seed is not None:
        s = seed
    else:
        s = int(idx[uncovered_arr][0])

    block = list(itertools.combinations(range(v), 5)[s]) if False else None
    # recover the actual t-subset from its lexicographic index
    # precompute map index->subset lazily
    block = list(_combo_from_index(s, v, 5))
    while len(block) < k:
        # gain(x) = number of uncovered t-subsets that contain x and the rest-in-block
        # For each uncovered t-subset, check if it contains x and (subset - x) subset of block
        best_x, best_gain = -1, -1
        for x in range(v):
            if x in block:
                continue
            # count uncovered subsets S where x in S and S\{x} ⊆ block
            # S\{x} is a (t-1)-subset of block
            cnt = 0
            for rest in itertools.combinations(block, 4):
                # build the 5-set rest + x, check if uncovered
                sset = tuple(sorted(rest + (x,)))
                j = _combo_index(sset, v, 5)
                if uncovered_arr[j]:
                    cnt += 1
            if cnt > best_gain:
                best_gain = cnt
                best_x = x
        block.append(best_x)
    block.sort()
    blocks.append(block)
    m = _cover_mask(block, 5, v)
    uncovered_arr &= ~m
    covered |= m
    return blocks, uncovered_arr


def _combo_from_index(idx, v, t):
    """Return the t-combination of range(v) at lexicographic index idx (unrank)."""
    # unrank algorithm
    n = v
    r = t
    result = []
    x = idx
    for i in range(t):
        # find smallest j such that C(n - j - 1, r - 1) <= x
        j = 0
        while True:
            c = _C(n - j - 1, r - 1)
            if c <= x:
                x -= c
                j += 1
            else:
                break
        result.append(j)
        n = n - j - 1
        r = r - 1
    return result


def _combo_index(combo, v, t):
    """Lexicographic index of a t-combination of range(v)."""
    # rank algorithm
    idx = 0
    n = v
    prev = -1
    for i, x in enumerate(combo):
        r = t - i
        # number of combos where this position picks a value in (prev+1 .. x-1)
        for j in range(prev + 1, x):
            idx += _C(n - j - 1, r - 1)
        n = n - x + prev + 1
        prev = x
    return idx


# Precompute binomial for small values
_C_table = {}
def _C(n, r):
    if r < 0 or r > n:
        return 0
    key = (n, r)
    if key in _C_table:
        return _C_table[key]
    # compute
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) // (i + 1)
    _C_table[key] = res
    return res


def _remove_redundant(blocks, v, t):
    """Drop blocks that cover no t-subset uniquely."""
    n = len(blocks)
    # for each t-subset, which blocks cover it
    # count coverage
    block_masks = [_cover_mask(b, t, v) for b in blocks]
    # count how many blocks cover each t-subset
    # use sum of masks bit-count per position via accumulation
    counts = np.zeros(_C(v, t), dtype=np.int32)
    for m in block_masks:
        # add 1 to each set bit position
        mm = m
        while mm:
            b = mm & (-mm)
            i = b.bit_length() - 1
            counts[i] += 1
            mm ^= b
    keep = []
    for i, m in enumerate(block_masks):
        # if any bit in m has counts==1, block is needed
        mm = m
        needed = False
        while mm:
            b = mm & (-mm)
            j = b.bit_length() - 1
            if counts[j] == 1:
                needed = True
                break
            mm ^= b
        if needed:
            keep.append(blocks[i])
    return keep


def construct(v, k, t):
    start = time.time()
    rng = random.Random(12345)
    total = _C(v, t)
    all_tsubs = list(itertools.combinations(range(v), t))
    idx_map = {c: i for i, c in enumerate(all_tsubs)}
    init_uncovered = [1 << i for i in range(total)]

    best = None
    best_len = float('inf')

    # Precompute combos for unrank
    # First run: deterministic greedy (seed = first uncovered)
    def greedy_from(uncovered_list, seed_idx):
        """Return list of blocks and remaining uncovered list."""
        n = len(uncovered_list)
        uncovered = list(uncovered_list)
        # build quick lookup: is index uncovered
        unc_set = set(uncovered)
        blocks = []
        remaining = seed_idx
        while remaining is not None and len(remaining) > 0:
            # pick seed subset
            s = remaining[0]
            block = list(all_tsubs[s])
            while len(block) < k:
                best_x, best_gain = -1, -1
                for x in range(v):
                    if x in block:
                        continue
                    cnt = 0
                    for rest in itertools.combinations(block, t - 1):
                        sset = tuple(sorted(rest + (x,)))
                        j = idx_map[sset]
                        if j in unc_set:
                            cnt += 1
                    if cnt > best_gain:
                        best_gain = cnt
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # remove all t-subsets in block
            m = _cover_mask(block, t, v)
            # remove from unc_set
            mm = m
            while mm:
                b = mm & (-mm)
                j = b.bit_length() - 1
                unc_set.discard(j)
                mm ^= b
            remaining = list(unc_set)
        return blocks

    # Run several randomized restarts
    for trial in range(2000):
        if time.time() - start > _TIME_LIMIT - 20:
            break
        # random seed subset
        seed_idx = rng.randrange(total)
        blocks = greedy_from(init_uncovered, [seed_idx])
        # redundancy removal
        blocks = _remove_redundant(blocks, v, t)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best = blocks
            # local search: try dropping each block and repairing
            improved = True
            while improved and time.time() - start < _TIME_LIMIT - 15:
                improved = False
                for bi in range(len(blocks)):
                    if time.time() - start > _TIME_LIMIT - 15:
                        break
                    # compute what would be uncovered if we drop blocks[bi]
                    # coverage counts
                    cnts = np.zeros(total, dtype=np.int32)
                    for b in blocks:
                        m = _cover_mask(b, t, v)
                        mm = m
                        while mm:
                            bb = mm & (-mm)
                            j = bb.bit_length() - 1
                            cnts[j] += 1
                            mm ^= bb
                    # drop bi
                    mbi = _cover_mask(blocks[bi], t, v)
                    mm = mbi
                    while mm:
                        bb = mm & (-mm)
                        j = bb.bit_length() - 1
                        cnts[j] -= 1
                        mm ^= bb
                    # uncovered now: cnts==0
                    unc = [j for j in range(total) if cnts[j] == 0]
                    if not unc:
                        # can drop
                        blocks = blocks[:bi] + blocks[bi + 1:]
                        improved = True
                        break
                    # repair: greedy on unc
                    unc_set = set(unc)
                    new_blocks = []
                    while unc_set and time.time() - start < _TIME_LIMIT - 10:
                        s = min(unc_set)
                        block = list(all_tsubs[s])
                        while len(block) < k:
                            best_x, best_gain = -1, -1
                            for x in range(v):
                                if x in block:
                                    continue
                                cnt = 0
                                for rest in itertools.combinations(block, t - 1):
                                    sset = tuple(sorted(rest + (x,)))
                                    j = idx_map[sset]
                                    if j in unc_set:
                                        cnt += 1
                                if cnt > best_gain:
                                    best_gain = cnt
                                    best_x = x
                            block.append(best_x)
                        block.sort()
                        new_blocks.append(block)
                        m = _cover_mask(block, t, v)
                        mm = m
                        while mm:
                            bb = mm & (-mm)
                            j = bb.bit_length() - 1
                            unc_set.discard(j)
                            mm ^= bb
                    candidate = blocks[:bi] + new_blocks + blocks[bi + 1:]
                    if len(candidate) < len(blocks):
                        blocks = candidate
                        improved = True
                        break
    if best is None:
        # fallback: one deterministic run
        best = greedy_from(init_uncovered, [0])
        best = _remove_redundant(best, v, t)
    return best
