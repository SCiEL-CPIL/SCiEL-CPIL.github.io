"""Greedy covering C(v,k,t) with vectorized BLAS-style gain scoring + restarts.

Design family (kept): randomized greedy set cover. The measured win was
vectorized gain scoring; here the per-point gain is a masked gather over a
precomputed index table (C-optimized), so many seeded restarts fit the deadline.
"""
import time
import itertools
import numpy as np

_DEADLINE_S = 115.0  # evaluator kills ~150s; keep a margin.


def _comb_count(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) // (i + 1)
    return res


def _comb_index(n, r, combo):
    idx = 0
    for i, x in enumerate(combo):
        idx += _comb_count(x, r - i - 1)
    return idx


def _tsub_from_index(idx, v, t):
    res = []
    n = v
    for i in range(t):
        r = t - i
        x = i
        while x < n and _comb_count(x + 1, r - 1) <= idx:
            x += 1
        res.append(x)
        idx -= _comb_count(x, r - 1)
        n = x
    return tuple(res)


def construct(v, k, t):
    start = time.time()
    n_t = _comb_count(v, t)

    tsub_index = {}
    for c in itertools.combinations(range(v), t):
        tsub_index[c] = _comb_index(v, t, c)

    # Precompute, for each (t-1)-subset S, add_index[si, x] = t-subset index of
    # S ∪ {x} (or -1 if x ∈ S).
    tminus1_subs = list(itertools.combinations(range(v), t - 1))
    tminus1_pos = {S: i for i, S in enumerate(tminus1_subs)}
    add_index = np.full((len(tminus1_subs), v), -1, dtype=np.int64)
    for si, S in enumerate(tminus1_subs):
        Sset = set(S)
        for x in range(v):
            if x in Sset:
                continue
            Sx = list(S)
            Sx.append(x)
            Sx.sort()
            add_index[si, x] = tsub_index[tuple(Sx)]

    def greedy_once(rng):
        uncovered = np.ones(n_t, dtype=np.uint8)
        blocks = []
        while uncovered.any():
            rem = np.flatnonzero(uncovered)
            seed = int(rem[rng.integers(len(rem))])
            tset = _tsub_from_index(seed, v, t)
            block = [int(tset[rng.integers(t)])]
            block_set = set(block)
            while len(block) < k:
                gains = np.zeros(v, dtype=np.int64)
                for S in itertools.combinations(block, t - 1):
                    ai = add_index[tminus1_pos[S]]
                    mask = ai >= 0
                    if not mask.any():
                        continue
                    idx = ai[mask]
                    vals = uncovered[idx]
                    np.add.at(gains, np.flatnonzero(mask), vals)
                gains[list(block_set)] = -1
                best_x = int(np.argmax(gains))
                if gains[best_x] <= 0:
                    avail = [x for x in range(v) if x not in block_set]
                    if not avail:
                        break
                    best_x = int(avail[rng.integers(len(avail))])
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered[tsub_index[c]] = 0
        return blocks

    best = None
    best_len = None
    rng = np.random.default_rng(0)
    while True:
        if time.time() - start > _DEADLINE_S - 3.0:
            break
        blocks = greedy_once(rng)
        if best_len is None or len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)
        if best_len is not None and best_len <= 131:
            break
        rng = np.random.default_rng(rng.integers(2**62))
    if best is None:
        best = greedy_once(np.random.default_rng(123))
    return best
