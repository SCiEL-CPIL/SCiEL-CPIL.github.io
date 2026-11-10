"""Vectorized greedy covering for C(19,7,4)."""
import time
import itertools
import numpy as np

# Precompute binomial coefficients (small v only)
def _build_comb_table(v):
    C = [[0] * (v + 1) for _ in range(v + 1)]
    for n in range(v + 1):
        C[n][0] = 1
        for r in range(1, n + 1):
            C[n][r] = C[n - 1][r - 1] + C[n - 1][r]
    return C

_COMB = _build_comb_table(64)

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 115.0  # safety margin under 120s

    n = v
    total = _COMB[v][t]

    # Rank table: rank[tuple(sorted subset)] -> index in [0, total)
    rank = {}
    idx = 0
    for combo in itertools.combinations(range(n), t):
        rank[combo] = idx
        idx += 1

    # Frequency of each point across the universe (for seed heuristic)
    freq = np.zeros(n, dtype=np.int64)
    for i in range(n):
        freq[i] = _COMB[n - 1][t - 1]

    # Uncovered universe mask
    uncovered = np.ones(total, dtype=np.uint8)
    point_set = set(range(n))

    # Precompute, for each point x, the list of (t-1)-subsets of the points
    # less than x (these combine with x to form t-subsets containing x).
    # Stored as flat arrays of length C(n-1, t-1).
    pre = {}
    for x in range(n):
        lst = [c for c in itertools.combinations(range(x), t - 1)]
        pre[x] = np.array(lst, dtype=np.int32)
    pre_len = _COMB[n - 1][t - 1]

    blocks = []
    best_valid = None
    best_count = 0

    def finalize():
        nonlocal best_valid, best_count
        # blocks is always valid (uncovered only shrinks); keep the longest valid prefix
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_valid = [list(b) for b in blocks]

    while True:
        if time.time() > DEADLINE:
            break

        # ---- choose a seed subset: first uncovered point in increasing freq order
        # (cheap; avoids O(total) scan). Try a few low-frequency points for variety.
        seed = None
        # order points by frequency ascending, break ties by index
        order = list(range(n))
        order.sort(key=lambda i: (freq[i], i))
        for i in order:
            if freq[i] == 0:
                continue
            # find one t-subset containing i that is uncovered
            # iterate t-1 subsets of the other points; check coverage
            others = [p for p in range(n) if p != i]
            found = None
            for sub in itertools.combinations(others, t - 1):
                full = tuple(sorted(sub + (i,)))
                if uncovered[rank[full]]:
                    found = list(full)
                    break
            if found is not None:
                seed = found
                break
        if seed is None:
            # fully covered
            break

        # ---- build the block greedily
        block = list(seed)
        block_set = set(block)

        # precompute (t-1)-subsets of the current block
        def block_t1subs(b):
            return list(itertools.combinations(b, t - 1))

        bsubs = block_t1subs(block)

        while len(block) < k:
            if time.time() > DEADLINE:
                break
            best_x = None
            best_gain = -1
            # for each candidate, count newly covered t-subsets
            # vectorize: for each x not in block, build its t-1 sub arrays and
            # combine with each block (t-1) subset to form t-subsets, rank them,
            # and count how many are uncovered.
            for x in range(n):
                if x in block_set:
                    continue
                # t-subsets containing x that are subsets of (block + x):
                # formed by choosing (t-1) from block and adding x
                # count uncovered among these
                gain = 0
                # use precomputed pre[x] (t-1 subs of points < x) and
                # also need (t-1) subs of (block) that may include points > x.
                # Simpler: enumerate (t-1) subsets of block and add x.
                # block size <= k-1 = 6, C(6,3)=20, small.
                for sub in bsubs:
                    full = tuple(sorted(sub + (x,)))
                    if uncovered[rank[full]]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                # no single point helps; add an arbitrary unused point to fill
                for x in range(n):
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                        bsubs = block_t1subs(block)
                        break
                else:
                    break
            else:
                block.append(best_x)
                block_set.add(best_x)
                bsubs = block_t1subs(block)

        block.sort()

        # apply coverage
        for sub in itertools.combinations(block, t):
            uncovered[rank[sub]] = 0
        # update frequencies
        for i in block:
            freq[i] -= _COMB[k - 1][t - 1]
            if freq[i] < 0:
                freq[i] = 0

        blocks.append(block)
        finalize()

        if not uncovered.any():
            break

    # Return best valid result
    if best_valid is not None:
        return best_valid
    return blocks
