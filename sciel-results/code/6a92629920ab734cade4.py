"""Greedy covering C(v,k,t) with ID-based gain, block-removal pass, deadline guard."""
import time
import itertools
import numpy as np


def _subset_id(combo, v, t):
    """Lexicographic rank of a sorted t-combo of range(v) in [0, C(v,t))."""
    n = 0
    for i in range(t):
        a = combo[i]
        r = t - i
        n += (a * (a - 1) * (a - 2) * (a - 3) * (a - 4) * (a - 5) // (120 * (r - 1))
              + (a - 1) * (a - 2) * (a - 3) * (a - 4) * (a - 5) // (24 * (r - 2))
              + (a - 2) * (a - 3) * (a - 4) * (a - 5) // (6 * (r - 3))
              + (a - 3) * (a - 4) * (a - 5) // (2 * (r - 4))
              + (a - 4) * (a - 5) // (r - 5))
    return n


def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 110.0  # safety margin under 120s

    all_tsub = list(itertools.combinations(range(v), t))
    ids = np.array([_subset_id(c, v, t) for c in all_tsub], dtype=np.int64)
    total_tsub = len(ids)

    uncovered = np.ones(total_tsub, dtype=bool)

    def gain_for(x, block):
        cnt = 0
        for s in itertools.combinations(block, t - 1):
            iid = _subset_id(tuple(sorted(s + (x,))), v, t)
            if uncovered[iid]:
                cnt += 1
        return cnt

    greedy_blocks = []
    while uncovered.any():
        if time.time() > DEADLINE:
            break
        first_idx = int(np.argmax(uncovered))
        block = list(all_tsub[first_idx])
        while len(block) < k:
            best_x, best_gain = -1, -1
            for x in range(v):
                if x in block:
                    continue
                g = gain_for(x, block)
                if g > best_gain:
                    best_gain, best_x = g, x
            if best_x < 0:
                break
            block.append(best_x)
        block.sort()
        for c in itertools.combinations(block, t):
            uncovered[_subset_id(c, v, t)] = False
        greedy_blocks.append(block)

    # Block-removal pass (safe reduction)
    def block_id_set(b):
        return set(_subset_id(c, v, t) for c in itertools.combinations(b, t))

    block_sets = [block_id_set(b) for b in greedy_blocks]
    counts = {}
    for bs in block_sets:
        for iid in bs:
            counts[iid] = counts.get(iid, 0) + 1

    keep = [True] * len(block_sets)
    for i, bs in enumerate(block_sets):
        if all(counts.get(iid, 0) >= 2 for iid in bs):
            keep[i] = False
            for iid in bs:
                counts[iid] -= 1

    reduced = [b for b, kk in zip(greedy_blocks, keep) if kk]

    # Verify reduced is still a valid cover; if not, use greedy_blocks
    if len(reduced) < len(greedy_blocks):
        cov = np.zeros(total_tsub, dtype=bool)
        for b in reduced:
            for c in itertools.combinations(b, t):
                cov[_subset_id(c, v, t)] = True
        if cov.all():
            return reduced
    return greedy_blocks
