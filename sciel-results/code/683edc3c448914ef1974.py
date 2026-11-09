import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

DEADLINE = 80.0


def _edge_set():
    return {(a, b) for a, b in P['edges']}


def _adj():
    adj = {i: set() for i in range(P['n'])}
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _initial_bicliques():
    # Star decomposition (matches seed): one biclique per vertex that has edges.
    adj = _adj()
    out = []
    for u in range(P['n']):
        if adj[u]:
            out.append([ [u], list(adj[u]) ])
    return out


def _biclique_edges(b, adj):
    L, R = b
    es = set()
    for u in L:
        for v in R:
            if v in adj[u]:
                es.add((u, v))
    return es


def _max_biclique_in_edge_set(es, adj):
    # Greedy maximal biclique inside the given edge set (both sides restricted to endpoints).
    if not es:
        return set(), set()
    # Start from the first edge.
    u0, v0 = next(iter(es))
    L = {u0}
    R = {v0}
    changed = True
    while changed:
        changed = False
        # Expand L: vertex u not in L that connects to all of R
        candL = set()
        for (a, b) in es:
            for x in (a, b):
                if x not in L:
                    candL.add(x)
        for u in list(candL):
            if all(v in adj[u] for v in R):
                L.add(u)
                changed = True
        # Expand R
        candR = set()
        for (a, b) in es:
            for x in (a, b):
                if x not in R:
                    candR.add(x)
        for v in list(candR):
            if all(u in adj[v] for u in L):
                R.add(v)
                changed = True
    # Ensure every cross pair is actually an edge in es (should hold by construction)
    L = {u for u in L if all(v in adj[u] and (u, v) in es for v in R)}
    R = {v for v in R if all(u in adj[v] and (u, v) in es for u in L)}
    return L, R


def _try_merge(bi, bj, es_i, es_j, adj, deadline):
    # Attempt to merge bicliques i and j into one valid biclique.
    # Candidate edge set = union of the two bicliques' cross edges.
    union = es_i | es_j
    if time.time() > deadline:
        return None
    L, R = _max_biclique_in_edge_set(union, adj)
    if not L or not R:
        return None
    # The merged biclique must cover exactly es_i | es_j (no more, no less)
    new_es = set()
    for u in L:
        for v in R:
            if (u, v) in union:
                new_es.add((u, v))
    if new_es != union:
        return None
    # Nonempty disjoint sides
    if L & R:
        return None
    return [list(L), list(R)]


def _greedy_reduce(bics, adj, deadline):
    # Repeatedly try to merge pairs of bicliques that can be combined into a valid biclique.
    # We keep the union-of-edges exactly covered.
    n = len(bics)
    if n < 2:
        return bics
    # Precompute edge sets
    es = [_biclique_edges(b, adj) for b in bics]
    # Sort by size to try merging small ones first (cheaper unions)
    order = list(range(n))
    order.sort(key=lambda i: len(es[i]))
    # Maintain active indices
    active = list(range(n))
    # Map active position -> original index
    # We'll just rebuild lists each pass.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Build current bicliques and edge sets
        cur_bics = [bics[i] for i in active]
        cur_es = [es[i] for i in active]
        m = len(cur_bics)
        if m < 2:
            break
        # Try all pairs
        found = False
        for a in range(m):
            if time.time() > deadline:
                break
            for b in range(a + 1, m):
                merged = _try_merge(cur_bics[a], cur_bics[b], cur_es[a], cur_es[b], adj, deadline)
                if merged is not None:
                    # Replace a,b with merged
                    new_bics = cur_bics[:a] + cur_bics[a+1:b] + [merged] + cur_bics[b+1:]
                    new_es = cur_es[:a] + cur_es[a+1:b] + [cur_es[a] | cur_es[b]] + cur_es[b+1:]
                    # Rebuild mapping to original indices
                    new_active = active[:a] + active[a+1:b] + [active[a]] + active[b+1:]
                    bics = new_bics
                    # Recompute es list fully (since indices changed)
                    es = [_biclique_edges(x, adj) for x in bics]
                    active = list(range(len(bics)))
                    improved = True
                    found = True
                    break
            if found:
                break
    return bics


def _local_search(bics, adj, deadline):
    # Local search: try merging any pair, and also try replacing two bicliques with a
    # single maximal biclique if it covers exactly their union. Same as greedy but with
    # random restarts / ordering for diversity.
    # For simplicity, run greedy a few times with different orderings.
    best = bics
    rng = random.Random(12345)
    for restart in range(5):
        if time.time() > deadline:
            break
        cur = [b[:] for b in bics]
        # Random shuffle of bicliques
        rng.shuffle(cur)
        cur = _greedy_reduce(cur, adj, deadline)
        if len(cur) < len(best):
            best = cur
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin
    adj = _adj()
    bics = _initial_bicliques()
    # Greedy reduction
    bics = _greedy_reduce(bics, adj, deadline)
    # Local search with restarts
    bics = _local_search(bics, adj, deadline)
    # Final validation: ensure partition covers all edges exactly once, each biclique valid.
    all_edges = _edge_set()
    covered = set()
    for b in bics:
        L, R = b
        assert len(L) > 0 and len(R) > 0
        assert not (set(L) & set(R))
        for u in L:
            for v in R:
                e = (u, v)
                assert e in all_edges, "cross pair not an edge"
                assert e not in covered, "edge covered twice"
                covered.add(e)
    assert covered == all_edges, "not all edges covered"
    # Convert to plain Python types
    result = [[list(map(int, L)), list(map(int, R))] for L, R in bics]
    return result
# EVOLVE-BLOCK-END
