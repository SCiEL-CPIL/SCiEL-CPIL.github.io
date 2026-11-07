"""Clique partition of three fixed split (chordal) graphs.
Keeps the exact instance() generator from the seed program.
Strategy: for each independent vertex i, partition its neighbor set S_i into
independent sets of the "already-claimed core edges" graph (via DSATUR), so each
part forms a valid cross clique {i}+part that covers i's cross edges and claims
the (still-free) core edges inside the part. Leftover core edges are covered by
a greedy max-clique edge cover on K. This packs cross edges into few large
cliques while keeping every edge in exactly one clique.
"""
import time

_DEADLINE = 110.0  # wall-clock safety margin (evaluator kills at 150s)


def _dsatur_parts(S, adj):
    """Partition S into independent sets of graph (S, adj) using DSATUR.
    Returns list of lists (the color classes)."""
    V = list(S)
    if not V:
        return []
    color = {}
    sat = {v: set() for v in V}
    uncolored = set(V)
    while uncolored:
        v = max(uncolored, key=lambda x: (len(sat[x]), len(adj.get(x, ()))) )
        c = 0
        while c in sat[v]:
            c += 1
        color[v] = c
        uncolored.discard(v)
        for w in adj.get(v, ()):
            if w in uncolored:
                sat[w].add(c)
    parts = {}
    for v, c in color.items():
        parts.setdefault(c, []).append(v)
    return [sorted(p) for p in parts.values()]


def _cover_leftover_core(F, K):
    """Cover the set F of core edges (on vertex set K) with a minimum-ish
    number of cliques (greedy maximal-clique removal). Returns list of cliques
    (each a sorted list of >=2 vertices)."""
    F = set(F)
    cliques = []
    while F:
        deg = {}
        for (a, b) in F:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        start = max(deg, key=lambda x: deg[x])
        clique = [start]
        changed = True
        while changed:
            changed = False
            cand = [v for v in K
                    if v not in clique
                    and all(frozenset((v, w)) in F for w in clique)]
            if cand:
                best = max(cand,
                           key=lambda v: sum(1 for w in clique
                                             if frozenset((v, w)) in F))
                clique.append(best)
                changed = True
        if len(clique) < 2:
            # should not happen (F nonempty implies an edge), but stay safe
            break
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                F.discard(frozenset((clique[a], clique[b])))
        cliques.append(sorted(clique))
    return cliques


def _solve_instance(n, k, p, seed):
    from random import Random
    rng = Random(seed)
    # cross adjacency: for each independent vertex i in [k, n), its K-neighbors
    S = {}
    for i in range(k, n):
        nbrs = [u for u in range(k) if rng.random() < p]
        S[i] = nbrs

    # all core edges
    core_edges = [frozenset((u, v)) for u in range(k) for v in range(u + 1, k)]

    claimed = set()
    cross_cliques = []

    # process independent vertices by decreasing neighbor-set size
    order = sorted(range(k, n), key=lambda i: -len(S[i]))
    for i in order:
        nbrs = S[i]
        if not nbrs:
            continue
        # build G_i on nbrs: edge (u,v) if core edge already claimed
        adj = {u: set() for u in nbrs}
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                if frozenset((nbrs[a], nbrs[b])) in claimed:
                    adj[nbrs[a]].add(nbrs[b])
                    adj[nbrs[b]].add(nbrs[a])
        parts = _dsatur_parts(nbrs, adj)
        for part in parts:
            if not part:
                continue
            cross_cliques.append([i] + part)
            # claim the (free) core edges inside this part
            for a in range(len(part)):
                for b in range(a + 1, len(part)):
                    claimed.add(frozenset((part[a], part[b])))

    # leftover core edges
    leftover = [e for e in core_edges if e not in claimed]
    core_cliques = _cover_leftover_core(set(leftover), list(range(k)))

    return cross_cliques + core_cliques


def construct():
    t0 = time.time()
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() - t0 > _DEADLINE:
            # safety: return what we have (will not normally trigger)
            break
        cliques = _solve_instance(*spec)
        # enforce exact types: list of lists of Python int
        cliques = [[int(v) for v in c] for c in cliques]
        result.append(cliques)
    return result
