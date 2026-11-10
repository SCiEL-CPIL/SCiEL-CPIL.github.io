# EVOLVE-BLOCK-START
import time
import random
from math import comb
from itertools import combinations
import numpy as np


def _combo_id(c, v, t):
    # Lexicographic rank of a sorted t-combo of {0..v-1} in [0, C(v,t))
    n = v
    r = t
    res = 0
    for i, x in enumerate(c):
        if x != i:
            res += comb(n - i - 1, r)
            n -= 1
            r -= 1
    return res


def _greedy(v, k, t, ids, uncovered, rng):
    n = v
    allc = list(combinations(range(n), t))
    block_counts = np.zeros(n, dtype=np.int64)
    blocks = []
    while True:
        rem = int(uncovered.sum())
        if rem == 0:
            break
        # Randomly pick an uncovered t-subset to seed the new block
        while True:
            c = allc[rng.randrange(len(allc))]
            if uncovered[ids[c]]:
                break
        block = [c[0]]
        used = np.zeros(n, dtype=bool)
        used[c[0]] = True
        while len(block) < k:
            B = block
            best = -1
            best_x = -1
            for x in range(n):
                if used[x]:
                    continue
                # gain = number of uncovered t-subsets formed by x with a (t-1)-subset of B
                tot = 0
                for sub in combinations(B, t - 1):
                    tot += int(uncovered[ids[sub + (x,)]])
                if tot > best:
                    best = tot
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            used[best_x] = True
            block_counts[best_x] += 1
        block.sort()
        blocks.append(tuple(block))
        for c in combinations(block, t):
            uncovered[ids[c]] = False
    return blocks


def construct(v, k, t):
    t0 = time.time()
    DEADLINE = t0 + 112.0   # safety margin before the 120s kill
    allc = list(combinations(range(v), t))
    ids = {c: _combo_id(c, v, t) for c in allc}
    base = np.ones(len(allc), dtype=bool)
    rng = random.Random(12345)

    best = None
    best_len = None
    while time.time() < DEADLINE:
        uncovered = base.copy()
        blocks = _greedy(v, k, t, ids, uncovered, rng)
        if best_len is None or len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)
            if time.time() > DEADLINE - 4.0:
                break
    return [list(b) for b in best]
# EVOLVE-BLOCK-END
