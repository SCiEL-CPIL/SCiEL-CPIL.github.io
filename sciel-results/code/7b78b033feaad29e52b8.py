"""Portfolio greedy covering design C(v,k,t) with NumPy acceleration."""
import itertools
import time
import numpy as np

# EVOLVE-BLOCK-START
def _make_covered_matrix(v, k, t):
    """Return (idx, covered) where covered is int8 array [n_tsets, v].
    covered[i, x] = 1 iff point x can cover t-set i (x not already in i)."""
    all_tsets = list(itertools.combinations(range(v), t))
    n = len(all_tsets)
    idx = np.empty((n, t), dtype=np.int8)
    for i, c in enumerate(all_tsets):
        idx[i] = c
    covered = np.ones((n, v), dtype=np.int8)
    # points already in each t-set cannot cover it
    covered[np.arange(n)[:, None], idx] = 0
    return idx, covered


def _greedy_cover(v, k, t, idx, covered, deadline):
    n = idx.shape[0]
    tsets = idx.copy()
    rem = np.ones(n, dtype=np.int8)

    # precompute: for each point x, which t-sets it could cover
    pt_tsets = [np.where(covered[:, x] == 1)[0] for x in range(v)]

    blocks = []
    while True:
        if time.time() >= deadline:
            break
        if not rem.any():
            break
        # choose rarest uncovered t-set (fewest covering points)
        uncovered_idx = np.where(rem == 1)[0]
        counts = covered[:, uncovered_idx].sum(axis=1)
        i = uncovered_idx[np.argmin(counts)]
        block = [int(tsets[i][0])]
        rem[i] = 0
        for _ in range(k - 1):
            if time.time() >= deadline:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                mask = rem[pt_tsets[x]] == 1
                g = int(mask.sum())
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            rem[pt_tsets[best_x]] = 0
        if len(block) >= 2:
            blocks.append(tuple(sorted(block)))
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 108.0  # safety margin before 120s kill
    idx, covered = _make_covered_matrix(v, k, t)
    blocks = _greedy_cover(v, k, t, idx, covered, deadline)

    # verify + patch any remaining uncovered t-sets with random blocks
    if blocks and time.time() < deadline - 3.0:
        covered_set = set()
        for b in blocks:
            for c in itertools.combinations(b, t):
                covered_set.add(c)
        all_tsets = list(itertools.combinations(range(v), t))
        uncovered = [c for c in all_tsets if c not in covered_set]
        while uncovered and time.time() < deadline - 1.0:
            c = uncovered.pop()
            block = list(c)
            for _ in range(k - t):
                block.append(np.random.randint(0, v))
            block = sorted(set(block))
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
            block = tuple(sorted(block[:k]))
            if block not in blocks:
                blocks.append(block)
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
