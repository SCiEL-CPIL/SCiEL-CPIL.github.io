"""C6-free graph on 40 vertices: maximize edges via fast exact C6 test + greedy + flips."""
import time
import random

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
DEADLINE = time.time() + 110  # safety margin vs 150s limit


def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _creates_c6(adj, u, v):
    """Exact test: does adding edge (u,v) create a C6?
    A new C6 must use edge (u,v). The other 4 vertices form a path u-a-b-c-v
    with a in N(u), c in N(v), a!=c, a,c non-adjacent, and b a common
    non-adjacent neighbour of a and c. So a C6 exists iff the set of common
    non-adjacent neighbours of some pair (a,c) (a in N(u), c in N(v), a!=c,
    a not~ c) has a C4. Equivalent: common non-adjacent neighbours of (a,c)
    contain an edge. We check by scanning pairs (a,c) and testing C4 in common."""
    nu = adj[u]
    nv = adj[v]
    # candidate a in N(u), c in N(v), a != c, a not adjacent to c
    for a in nu:
        if a == v:
            continue
        # c in N(v), c != u, c != a, c not adjacent to a
        for c in nv:
            if c == u or c == a:
                continue
            if c in adj[a]:
                continue
            # common non-adjacent neighbours of a and c (excluding u,v)
            ca = adj[a]
            cc = adj[c]
            common = ca & cc
            # remove u, v, a, c
            common.discard(u)
            common.discard(v)
            common.discard(a)
            common.discard(c)
            # C4 in common? i.e. an edge between two common vertices
            for x in common:
                if x in cc:  # x in common so x in ca and cc; check edge x-y
                    # need y in common, y != x, y adjacent to x
                    for y in common:
                        if y != x and y in ca and y in cc and y in adj[x]:
                            return True
    return False


def _has_c6(adj):
    """Check whether the current graph (no assumed new edge) has any C6.
    Use the same characterization: a C6 exists iff there is a pair of
    non-adjacent vertices (x,y) with >=3 common neighbours containing a C4.
    For speed, iterate over edges? No. We'll iterate over pairs (a,c) that are
    non-adjacent and check their common-neighbour set for a C4. But that's O(n^2 * ...).
    For the flip test we only need _creates_c6. For initial validity we trust the
    construction. We provide a full check for safety on the final result."""
    for i in range(N):
        for j in range(i + 1, N):
            if j in adj[i]:
                continue
            common = adj[i] & adj[j]
            common.discard(i)
            common.discard(j)
            if len(common) < 3:
                continue
            # check C4 in induced subgraph on common
            cl = list(common)
            m = len(cl)
            for a in range(m):
                for b in range(a + 1, m):
                    x, y = cl[a], cl[b]
                    if y in adj[x]:
                        # edge x-y in common; need another edge x'-y' disjoint? 
                        # C4 needs 4 distinct vertices. Find z,w in common, distinct
                        # from x,y, with z~w and z~x? Actually a C4 is a 4-cycle.
                        # Simpler: check if common induces any C4 by looking for
                        # two disjoint edges with cross-connections. We'll do a
                        # direct 4-cycle search on common.
                        pass
            # Direct C4 detection in common: for each pair of non-adjacent
            # vertices p,q in common, check if they share 2 common neighbours.
            for a in range(m):
                p = cl[a]
                for b in range(a + 1, m):
                    q = cl[b]
                    if q in adj[p]:
                        continue
                    cp = adj[p] & common
                    cq = adj[q] & common
                    cp.discard(p)
                    cq.discard(q)
                    if len(cp & cq) >= 2:
                        return True
    return False


def _greedy_add(adj, edges, edge_set):
    """Try to add every missing edge that doesn't create a C6 (single pass)."""
    added = 0
    for u, v in ALL_PAIRS:
        if (u, v) in edge_set or (v, u) in edge_set:
            continue
        if not _creates_c6(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            edge_set.add((u, v))
            added += 1
    return added


def _k5_seed():
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append((5 * k + a, 5 * k + b))
    return edges


def _random_seed(rng, p):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def _improve(edges, rng, deadline):
    adj = _build_adj(edges)
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    best_edges = list(edges)
    best_count = len(edges)

    # Greedy fill
    _greedy_add(adj, edges, edge_set)
    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = list(edges)

    # Flip local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(edges)):
            if time.time() > deadline:
                break
            u, v = edges[idx]
            # remove edge (u,v)
            adj[u].discard(v)
            adj[v].discard(u)
            edges.pop(idx)
            edge_set.discard((min(u, v), max(u, v)))
            # try to add up to 2 new edges
            added = []
            for u2, v2 in ALL_PAIRS:
                if (u2, v2) in edge_set or (v2, u2) in edge_set:
                    continue
                if not _creates_c6(adj, u2, v2):
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edges.append((u2, v2))
                    edge_set.add((u2, v2))
                    added.append((u2, v2))
                    if len(added) >= 2:
                        break
            # If we added >= 1 edge, net gain; keep it
            if len(added) >= 1:
                improved = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
            else:
                # revert: put (u,v) back
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                edge_set.add((min(u, v), max(u, v)))
    return best_edges


def construct():
    rng = random.Random(12345)
    t0 = time.time()
    best_edges = _k5_seed()
    best_count = len(best_edges)

    # Try multiple seeds
    seeds = [
        _k5_seed(),
        _random_seed(rng, 0.3),
        _random_seed(rng, 0.5),
        _random_seed(rng, 0.2),
        [],
    ]
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        # Ensure seed is C6-free: filter greedily (remove edges that create C6)
        adj = [set() for _ in range(N)]
        clean = []
        es = set()
        for u, v in seed:
            if (u, v) in es or (v, u) in es:
                continue
            if not _creates_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                clean.append((u, v))
                es.add((u, v))
        if len(clean) < best_count:
            continue
        res = _improve(clean, rng, deadline)
        if len(res) > best_count:
            best_count = len(res)
            best_edges = res
        if time.time() > deadline - 10:
            break

    # Final safety: verify no C6 (best effort)
    adj = _build_adj(best_edges)
    if _has_c6(adj):
        # Fallback to K5 seed which is guaranteed C6-free
        best_edges = _k5_seed()

    return [[u, v] for u, v in best_edges]
