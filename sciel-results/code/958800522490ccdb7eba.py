# EVOLVE-BLOCK-START
import itertools
import time
import random
from math import comb

def _greedy_cover(v, k, t, rng, deadline):
    """Greedy set cover: open a block on an uncovered t-subset, grow to size k."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        if time.time() > deadline:
            break
        # Pick a seed t-subset (random for diversity)
        start = rng.choice(tuple(uncovered))
        block = list(start)
        while len(block) < k:
            if time.time() > deadline:
                break
            # Precompute (t-1)-subsets of current block
            rests = list(itertools.combinations(block, t - 1))
            best_x = None
            best_gain = -1
            best_tie = None
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in rests:
                    tri = tuple(sorted(rest + (x,)))
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tie = 1
                elif gain == best_gain and gain > 0:
                    if best_tie is None:
                        best_tie = 1
                    else:
                        best_tie += 1
                        if rng.random() < (1.0 / best_tie):
                            best_x = x
            if best_x is None:
                break
            block.append(best_x)
        block.sort()
        blocks.append(block)
        for tri in itertools.combinations(block, t):
            uncovered.discard(tri)
    return blocks


def _is_valid(blocks, v, k, t):
    total = comb(v, t)
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered) == total


def _local_search(blocks, v, k, t, rng, deadline):
    """Reduce block count via removal and single-block replacement."""
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    blocks = [sorted(list(b)) for b in blocks]

    # coverage count per t-subset
    cov = [0] * n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] += 1

    def recompute_cov():
        nonlocal cov
        cov = [0] * n_t
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[t_index[ts]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        order = list(range(n))
        rng.shuffle(order)
        for bi in order:
            if time.time() > deadline:
                return blocks
            if bi >= len(blocks):
                continue
            b = blocks[bi]
            # t-subsets uniquely covered by b
            unique = []
            for ts in itertools.combinations(b, t):
                idx = t_index[ts]
                if cov[idx] == 1:
                    unique.append(idx)
            if not unique:
                # b fully redundant -> remove
                blocks.pop(bi)
                recompute_cov()
                improved = True
                break
            # union of points in unique t-subsets
            cand = set()
            for idx in unique:
                cand.update(all_t[idx])
            cand = list(cand)
            if len(cand) < k:
                continue
            # If union fits in k, we can build a replacement containing all unique points.
            # Fill remaining spots to maximize rescued unique coverage.
            if len(cand) <= k:
                newblock = sorted(cand)
                # fill remaining
                while len(newblock) < k:
                    in_block = set(newblock)
                    best_x = None
                    best_gain = -1
                    for x in range(v):
                        if x in in_block:
                            continue
                        gain = 0
                        rests = list(itertools.combinations(newblock, t - 1))
                        for rest in rests:
                            tri = tuple(sorted(rest + (x,)))
                            idx = t_index.get(tri)
                            if idx is not None and cov[idx] >= 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    newblock.append(best_x)
                newblock.sort()
                if newblock != b and newblock not in blocks:
                    blocks.pop(bi)
                    blocks.append(newblock)
                    recompute_cov()
                    improved = True
                    break
            else:
                # union too large to fit in one block; try a greedy k-subset of cand
                # that covers as many unique as possible
                remaining = set(unique)
                newblock = []
                for _ in range(k):
                    if not remaining:
                        break
                    best_x = None
                    best_c = -1
                    for x in cand:
                        if x in newblock:
                            continue
                        c = 0
                        for idx in remaining:
                            if x in all_t[idx]:
                                c += 1
                        if c > best_c:
                            best_c = c
                            best_x = x
                    if best_x is None or best_c <= 0:
                        break
                    newblock.append(best_x)
                    to_remove = set()
                    for idx in remaining:
                        if all(p in newblock for p in all_t[idx]):
                            to_remove.add(idx)
                    remaining -= to_remove
                if not remaining and len(newblock) == k:
                    newblock.sort()
                    if newblock != b and newblock not in blocks:
                        blocks.pop(bi)
                        blocks.append(newblock)
                        recompute_cov()
                        improved = True
                        break
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0  # safety margin before 120s kill
    best_blocks = None
    best_len = None
    rng = random.Random(12345)

    # Baseline greedy + local search
    b0 = _greedy_cover(v, k, t, rng, deadline)
    if _is_valid(b0, v, k, t):
        best_blocks = b0
        best_len = len(b0)

    if best_blocks is not None and time.time() < deadline:
        ls = _local_search(best_blocks, v, k, t, rng, deadline)
        if _is_valid(ls, v, k, t) and len(ls) < best_len:
            best_blocks = ls
            best_len = len(ls)

    # Randomized restarts with local search
    attempt = 0
    while time.time() < deadline - 15.0:
        attempt += 1
        rng2 = random.Random(1000 + attempt)
        b = _greedy_cover(v, k, t, rng2, deadline - 12.0)
        if not _is_valid(b, v, k, t):
            continue
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
        if time.time() < deadline - 8.0:
            ls = _local_search(b, v, k, t, rng2, deadline - 8.0)
            if _is_valid(ls, v, k, t) and len(ls) < best_len:
                best_len = len(ls)
                best_blocks = ls

    if best_blocks is None:
        b = _greedy_cover(v, k, t, random.Random(1), time.time() + 5)
        return b
    return best_blocks
# EVOLVE-BLOCK-END
