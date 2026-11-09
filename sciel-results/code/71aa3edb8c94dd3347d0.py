"""Biclique partition: greedy set-cover over candidate bicliques."""
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = int(P['n'])
EDGES = []
for (a, b) in P['edges']:
    a = int(a); b = int(b)
    if a == b:
        continue
    if a > b:
        a, b = b, a
    EDGES.append((a, b))
M = len(EDGES)
EDGE_SET = set(EDGES)


def build_adj():
    adj = [set() for _ in range(N)]
    for (a, b) in EDGES:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def star_bicliques(adj):
    """For each vertex u, the largest biclique with u on the left."""
    cands = []
    for u in range(N):
        nbrs = sorted(adj[u])
        if not nbrs:
            continue
        R = nbrs[:]
        for w in nbrs:
            for v in list(R):
                if v == w:
                    continue
                if v not in adj[w]:
                    R.remove(v)
        if R:
            cands.append(([u], R))
    return cands


def greedy_select(cands, adj):
    """Greedy set-cover: pick the biclique covering the most uncovered edges."""
    remaining = set(EDGES)
    selected = []
    if not remaining:
        return selected
    cands = [c for c in cands if c[0] and c[1]]
    while remaining:
        best = None
        best_cov = -1
        for (L, R) in cands:
            cov = 0
            for u in L:
                for v in R:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in remaining:
                        cov += 1
            if cov > best_cov:
                best_cov = cov
                best = (L, R)
        if best is None or best_cov <= 0:
            # safety: cover remaining edges as single-edge bicliques
            for (a, b) in sorted(remaining):
                selected.append(([a], [b]))
            break
        L, R = best
        selected.append((L, R))
        for u in L:
            for v in R:
                a, b = (u, v) if u < v else (v, u)
                remaining.discard((a, b))
    return selected


def fallback_partition(adj):
    """Guaranteed-valid partition: one biclique per vertex."""
    out = []
    for u in range(N):
        nbrs = sorted(adj[u])
        if nbrs:
            out.append(([u], nbrs))
    return out


def validate(partition):
    seen = set()
    for (L, R) in partition:
        if not L or not R:
            return False
        for u in L:
            if u in R:
                return False
        for u in L:
            for v in R:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in EDGE_SET:
                    return False
                if (a, b) in seen:
                    return False
                seen.add((a, b))
    return seen == EDGE_SET


def construct():
    t0 = time.time()
    deadline = t0 + 60.0
    adj = build_adj()
    rng = random.Random(12345)

    base_cands = star_bicliques(adj)
    best = None

    # Primary: greedy on star bicliques
    part = greedy_select(base_cands, adj)
    if validate(part):
        best = part

    # Try additional candidate families within remaining budget
    attempts = 0
    while time.time() < deadline - 1.0 and attempts < 8:
        attempts += 1
        # Build extra candidates: random subsets of star bicliques (still valid bicliques)
        extra = []
        for (L, R) in base_cands:
            if len(R) > 1 and rng.random() < 0.6:
                k = rng.randint(1, len(R))
                Rsub = rng.sample(R, k)
                extra.append(([L[0]], Rsub))
        cands = base_cands + extra
        part = greedy_select(cands, adj)
        if validate(part) and (best is None or len(part) < len(best)):
            best = part

    if best is None:
        best = fallback_partition(adj)

    # Ensure exact types: list of [list, list] of Python ints
    out = []
    for (L, R) in best:
        out.append(([int(x) for x in L], [int(x) for x in R]))
    return out
