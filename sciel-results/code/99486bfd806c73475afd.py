"""Biclique partition via greedy largest-biclique selection."""
import time
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_DEADLINE = 80.0  # wall-clock seconds; keep margin under the 90s kill

def _grow_biclique(start_u, start_v, adj, remaining):
    """Grow a biclique from a seed edge (start_u, start_v) using remaining edges.
    Returns (L, R, covered_set) where L is left vertex list, R right vertex list,
    and covered_set is the frozenset of edges (as frozensets) that would be covered."""
    L = {start_u}
    R = {start_v}
    # Iterate to a fixed point: expand L to all vertices connected to every R,
    # expand R to all vertices connected to every L, intersected with remaining.
    changed = True
    while changed:
        changed = False
        # New L candidates: vertices u' such that (u', v) in remaining for all v in R
        new_L = set()
        for u_cand in range(P['n']):
            if u_cand in L:
                continue
            ok = True
            for v in R:
                if frozenset((u_cand, v)) not in remaining:
                    ok = False
                    break
            if ok:
                new_L.add(u_cand)
        if new_L:
            L |= new_L
            changed = True
        # New R candidates: vertices v' such that (u, v') in remaining for all u in L
        new_R = set()
        for v_cand in range(P['n']):
            if v_cand in R:
                continue
            ok = True
            for u in L:
                if frozenset((u, v_cand)) not in remaining:
                    ok = False
                    break
            if ok:
                new_R.add(v_cand)
        if new_R:
            R |= new_R
            changed = True
    # Covered edges: all (u, v) in L x R that are in remaining
    covered = set()
    for u in L:
        for v in R:
            e = frozenset((u, v))
            if e in remaining:
                covered.add(e)
    return list(L), list(R), covered

def construct():
    t0 = time.time()
    adj = defaultdict(set)
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)

    # Represent remaining edges as a set of frozensets of size 2
    remaining = set()
    for a, b in P['edges']:
        remaining.add(frozenset((a, b)))

    bicliques = []
    while remaining:
        if time.time() - t0 > _DEADLINE:
            # Safety: fall back to per-edge bicliques for remaining
            break
        # Find the largest biclique by trying each remaining edge as a seed.
        # To save time, only try a limited number of seeds per iteration.
        best_L, best_R, best_cov = None, None, None
        best_size = 0
        # Iterate over remaining edges; pick the one whose greedy growth is largest.
        # To bound runtime, sample if too many.
        edges_list = list(remaining)
        # Try all if small, else sample
        if len(edges_list) <= 200:
            seeds = edges_list
        else:
            # sample
            import random
            random.seed(12345)
            seeds = random.sample(edges_list, 200)
        for e in seeds:
            u = min(e)
            v = max(e)
            L, R, cov = _grow_biclique(u, v, adj, remaining)
            if len(cov) > best_size:
                best_size = len(cov)
                best_L, best_R, best_cov = L, R, cov
        if best_cov is None or len(best_cov) == 0:
            # Should not happen; fallback single edge
            e = next(iter(remaining))
            u = min(e)
            v = max(e)
            best_L, best_R = [u], [v]
            best_cov = {e}
        bicliques.append([list(best_L), list(best_R)])
        remaining -= best_cov

    # Fallback: if deadline hit, add each remaining edge as its own biclique
    for e in remaining:
        u = min(e)
        v = max(e)
        bicliques.append([int(u), int(v)] if False else [[int(u)], [int(v)]])

    # Ensure types: lists of Python ints
    out = []
    for L, R in bicliques:
        out.append([[int(x) for x in L], [int(x) for x in R]])
    return out
