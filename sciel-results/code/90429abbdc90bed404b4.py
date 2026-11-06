import time
import random
import numpy as np
from itertools import combinations

N = 16
K = 4
C = 1820  # C(16,4)
DEADLINE = 85.0

# Precompute every 4-subset of [1..16] as a 16-bit mask.
_ALL = []
for a in range(1, 17):
    for b in range(a + 1, 17):
        for c in range(b + 1, 17):
            for d in range(c + 1, 17):
                _ALL.append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (d - 1)))
M = np.array(_ALL, dtype=np.uint32)
assert len(M) == C

# A "sunflower" triple = three distinct 4-sets whose three pairwise intersections are all equal.
# For a 4-set the only possible equal-intersection sizes are 0 (empty core) and 1 (1-core).
# 2- and 3-cores are geometrically impossible for three distinct 4-sets.
def _triples():
    t = []
    for i in range(C):
        mi = M[i]
        for j in range(i + 1, C):
            mij = mi & M[j]
            for k in range(j + 1, C):
                mij2 = mi & M[k]
                if mij2 != mij:
                    continue
                if (M[j] & M[k]) != mij:
                    continue
                t.append((i, j, k))
    return t

_TRIPLES = _triples()

def _valid(fam):
    for (i, j, k) in _TRIPLES:
        if fam[i] and fam[j] and fam[k]:
            return False
    return True

def _greedy(order, fam):
    for idx in order:
        mi = M[idx]
        for (i, j, k) in _TRIPLES:
            if (i == idx or j == idx or k == idx) and fam[i] and fam[j] and fam[k]:
                break
        else:
            fam[idx] = 1
            continue
        break

def _local_search(fam, rng, budget):
    best = int(fam.sum())
    while budget() > 0:
        # random add
        idx = rng.randrange(C)
        if not fam[idx]:
            ok = True
            for (i, j, k) in _TRIPLES:
                if (i == idx or j == idx or k == idx) and fam[i] and fam[j] and fam[k]:
                    ok = False
                    break
            if ok:
                fam[idx] = 1
                best = max(best, int(fam.sum()))
                continue
        # random removal (repair)
        idx = rng.randrange(C)
        if fam[idx]:
            fam[idx] = 0
    return best

def _run_one(rng):
    order = list(range(C))
    rng.shuffle(order)
    fam = np.zeros(C, dtype=np.uint8)
    _greedy(order, fam)
    best = int(fam.sum())
    budget = lambda: time.time() < DEADLINE
    best = _local_search(fam, rng, budget)
    return best

def construct():
    start = time.time()
    rng = random.Random(12345)
    best = 0
    while time.time() < DEADLINE:
        best = max(best, _run_one(rng))
        # deterministic baseline on first pass
    # ensure at least a valid baseline exists
    if best <= 0:
        fam = np.zeros(C, dtype=np.uint8)
        _greedy(list(range(C)), fam)
        best = int(fam.sum())
    return best
