import time
import itertools
import random

N = 20
C = 20
ALL_PAIRS = list(itertools.combinations(range(N), 2))
M = len(ALL_PAIRS)

_pidx = {}
for i, (u, v) in enumerate(ALL_PAIRS):
    _pidx[(u, v)] = i

# Precompute 4-cycles: for each pair, list of 4-cycles (frozenset of 4 pair-indices) containing it
_cycles_per_pair = [[] for _ in range(M)]
_seen = set()
for i, (a, b) in enumerate(ALL_PAIRS):
    for c, d in itertools.combinations(range(N), 2):
        if c in (a, b) or d in (a, b):
            continue
        e1 = tuple(sorted((b, c)))
        e2 = tuple(sorted((c, d)))
        e3 = tuple(sorted((d, a)))
        j1 = _pidx[e1]
        j2 = _pidx[e2]
        j3 = _pidx[e3]
        cyc = frozenset((i, j1, j2, j3))
        if cyc in _seen:
            continue
        _seen.add(cyc)
        for p in (i, j1, j2, j3):
            _cycles_per_pair[p].append(cyc)

def construct():
    start = time.time()
    deadline = start + 130.0

    present = [False] * M
    color = [-1] * M

    def is_valid_add(i, c):
        # Check all 4-cycles containing i: if the other 3 are present,
        # their colours must be distinct from each other and from c.
        for cyc in _cycles_per_pair[i]:
            others = [p for p in cyc if p != i]
            present_others = [p for p in others if present[p]]
            if len(present_others) == 3:
                cols = [color[p] for p in present_others]
                if c in cols:
                    return False
                if len(set(cols)) != 3:
                    return False
        return True

    def add(i, c):
        present[i] = True
        color[i] = c

    # Greedy: process pairs in order of decreasing 4-cycle count (most constrained first)
    order = sorted(range(M), key=lambda i: -len(_cycles_per_pair[i]))

    for i in order:
        if time.time() > deadline - 5.0:
            break
        # Try colours in a deterministic order
        for c in range(C):
            if is_valid_add(i, c):
                add(i, c)
                break

    # Second pass: try remaining in random orders
    for attempt in range(8):
        if time.time() > deadline - 5.0:
            break
        random.seed(attempt * 13 + 3)
        remaining = [i for i in range(M) if not present[i]]
        random.shuffle(remaining)
        for i in remaining:
            if time.time() > deadline - 5.0:
                break
            for c in range(C):
                if is_valid_add(i, c):
                    add(i, c)
                    break

    edges = []
    for i, (u, v) in enumerate(ALL_PAIRS):
        if present[i]:
            edges.append([u, v, color[i]])
    return edges
