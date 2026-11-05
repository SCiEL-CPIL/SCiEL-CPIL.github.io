"""Covering design C(v,k,t) constructor.
Greedy construction + bitset coverage + iterative local-search improvement
(redundancy deletion + single point-swap), with randomized restarts and a
hard wall-clock deadline guard.
"""
import time
import random
import itertools
import numpy as np

# Precomputed combinatorial-rank table for v up to 40 (fast, one-time).
_VMAX = 40
_N = [[0] * (_VMAX + 1) for _ in range(_VMAX + 1)]
for _n in range(1, _VMAX + 1):
    _N[_n][0] = 1
    for _k in range(1, _n + 1):
        _N[_n][_k] = _N[_n - 1][_k] + _N[_n - 1][_k - 1]


def _rank(x, r, n):
    """Combinatorial number system rank of combination x (sorted, len r, from n)."""
    s = 0
    for i, v in enumerate(x):
        s += _N[v][r - 1 - i]
    return s


class _Cov:
    def __init__(self, v, k, t):
        self.v = v
        self.k = k
        self.t = t
        self.total = _N[v][t]
        self.covered = np.zeros(self.total, dtype=bool)
        self.cnt = np.zeros(self.total, dtype=np.int32)
        self.ncovered = 0

    def add(self, block):
        c = self.cnt
        cov = self.covered
        for ts in itertools.combinations(block, self.t):
            r = _rank(ts, self.t, self.v)
            c[r] += 1
            if not cov[r]:
                cov[r] = True
                self.ncovered += 1

    def remove(self, block):
        c = self.cnt
        cov = self.covered
        for ts in itertools.combinations(block, self.t):
            r = _rank(ts, self.t, self.v)
            c[r] -= 1
            if c[r] == 0:
                cov[r] = False
                self.ncovered -= 1

    def is_redundant(self, block):
        c = self.cnt
        for ts in itertools.combinations(block, self.t):
            if c[_rank(ts, self.t, self.v)] <= 1:
                return False
        return True

    def unique_count(self, block):
        c = self.cnt
        m = 0
        for ts in itertools.combinations(block, self.t):
            if c[_rank(ts, self.t, self.v)] == 1:
                m += 1
        return m


def _greedy(v, k, t, rng):
    cov = _Cov(v, k, t)
    blocks = []
    while cov.ncovered < cov.total:
        # pick the seed t-set with the fewest covering blocks (most constrained)
        minc = 10 ** 9
        seed = None
        for ts in itertools.combinations(range(v), t):
            cc = cov.cnt[_rank(ts, t, v)]
            if cc < minc:
                minc = cc
                seed = list(ts)
                if cc == 0:
                    break
        block = list(seed)
        inb = [False] * v
        for x in block:
            inb[x] = True
        while len(block) < k:
            best = -1
            bestx = -1
            for x in range(v):
                if inb[x]:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    rr = _rank(tuple(sorted(rest + (x,))), t, v)
                    if not cov.covered[rr]:
                        gain += 1
                if gain > best:
                    best = gain
                    bestx = x
            if bestx < 0:
                break
            block.append(bestx)
            inb[bestx] = True
        block.sort()
        cov.add(block)
        blocks.append(block)
    return blocks


def _improve(blocks, cov, v, k, t, rng, deadline):
    # Phase 1: delete redundant blocks (fixed point)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                return blocks, cov
            if cov.is_redundant(blocks[i]):
                cov.remove(blocks[i])
                del blocks[i]
                changed = True

    # Phase 2: single point-swap moves
    progress = True
    while progress:
        progress = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks, cov
            bi = blocks[i]
            for ri in range(k):
                for x in range(v):
                    if x in bi:
                        continue
                    newb = bi[:ri] + [x] + bi[ri + 1:]
                    cov.remove(bi)
                    u = cov.unique_count(newb)
                    cov.add(newb)
                    if u < cov.unique_count(bi):
                        blocks[i] = newb
                        progress = True
                    else:
                        cov.remove(newb)
                        cov.add(bi)
    return blocks, cov


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    rng = random.Random(12345)

    best = None
    bestn = None

    while time.time() < deadline:
        blocks = _greedy(v, k, t, rng)
        cov = _Cov(v, k, t)
        for b in blocks:
            cov.add(b)
        if cov.ncovered < cov.total:
            continue
        blocks, cov = _improve(blocks, cov, v, k, t, rng, deadline)
        if cov.ncovered < cov.total:
            continue
        n = len(blocks)
        if bestn is None or n < bestn:
            bestn = n
            best = blocks
            if bestn <= 131:
                break
        # perturbation: drop a few random blocks and re-greedy-fill
        if bestn is not None and bestn > 131:
            seed_blocks = [list(b) for b in best]
            if len(seed_blocks) > 2:
                m = max(1, len(seed_blocks) // 20)
                for _ in range(m):
                    if seed_blocks:
                        del seed_blocks[rng.randrange(len(seed_blocks))]
                cv = _Cov(v, k, t)
                for b in seed_blocks:
                    cv.add(b)
                while cv.ncovered < cv.total and time.time() < deadline:
                    minc = 10 ** 9
                    seed = None
                    for ts in itertools.combinations(range(v), t):
                        cc = cv.cnt[_rank(ts, t, v)]
                        if cc < minc:
                            minc = cc
                            seed = list(ts)
                            if cc == 0:
                                break
                    block = list(seed)
                    inb = [False] * v
                    for x in block:
                        inb[x] = True
                    while len(block) < k:
                        bestg = -1
                        bestx = -1
                        for x in range(v):
                            if inb[x]:
                                continue
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                rr = _rank(tuple(sorted(rest + (x,))), t, v)
                                if not cv.covered[rr]:
                                    gain += 1
                            if gain > bestg:
                                bestg = gain
                                bestx = x
                        if bestx < 0:
                            break
                        block.append(bestx)
                        inb[bestx] = True
                    block.sort()
                    cv.add(block)
                    seed_blocks.append(block)
                if cv.ncovered == cv.total:
                    seed_blocks, cv = _improve(seed_blocks, cv, v, k, t, rng, deadline)
                    if cv.ncovered == cv.total and len(seed_blocks) < bestn:
                        bestn = len(seed_blocks)
                        best = seed_blocks

    if best is None:
        # safety net: plain greedy (should always be valid)
        blocks = _greedy(v, k, t, rng)
        cov = _Cov(v, k, t)
        for b in blocks:
            cov.add(b)
        best = blocks
    return best
