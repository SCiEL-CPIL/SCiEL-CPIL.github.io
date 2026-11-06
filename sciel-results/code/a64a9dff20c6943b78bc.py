"""Clique partition of two dense random graphs via complement-graph independent-set greedy."""
import time
import random as _rnd

def _gen_graph(n, p, seed):
    rng = _rnd.Random(seed)
    # adjacency as set of frozensets
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges

def _max_clique_greedy(adj, n, uncovered_edges, start_vertex=None):
    """Greedy grow a clique in the original graph.
    uncovered_edges: set of (u,v) still needing cover.
    Returns a clique (list of vertices) that is a clique in original and covers >=1 uncovered edge.
    We grow by intersecting neighborhoods, preferring vertices that add many uncovered edges.
    """
    # candidate vertices that are endpoints of uncovered edges
    cand = set()
    for (u, v) in uncovered_edges:
        cand.add(u)
        cand.add(v)
    if not cand:
        return None

    # pick a starting vertex: one incident to many uncovered edges
    if start_vertex is not None and start_vertex in cand:
        cur = [start_vertex]
    else:
        # degree in uncovered-edge subgraph
        deg = {x: 0 for x in cand}
        for (u, v) in uncovered_edges:
            deg[u] += 1
            deg[v] += 1
        start = max(cand, key=lambda x: (deg[x], -x))
        cur = [start]

    cur_set = set(cur)
    # intersection of neighborhoods of cur
    common = set(adj[cur[0]])
    for x in cur[1:]:
        common &= adj[x]
    common -= cur_set

    # iterate: add vertex from common that maximizes new uncovered edges covered
    while True:
        if not common:
            break
        # for each candidate, count uncovered edges it would cover with cur
        best = None
        best_gain = 0
        for x in common:
            gain = 0
            for y in cur:
                e = (min(x, y), max(x, y))
                if e in uncovered_edges:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = x
            elif gain == best_gain and best is not None and x < best:
                best = x
        if best is None or best_gain <= 0:
            break
        cur.append(best)
        cur_set.add(best)
        common &= adj[best]
        common -= cur_set

    # keep only if it covers at least one uncovered edge
    covered = 0
    for i in range(len(cur)):
        for j in range(i + 1, len(cur)):
            e = (min(cur[i], cur[j]), max(cur[i], cur[j]))
            if e in uncovered_edges:
                covered += 1
    if covered == 0:
        return None
    return cur

def _partition_instance(adj, edges, deadline):
    n = len(adj)
    uncovered = set(edges)
    cliques = []
    while uncovered:
        if time.time() > deadline:
            # fallback: cover remaining as pairs (each pair is a 2-clique in original iff edge exists)
            for (u, v) in uncovered:
                cliques.append([u, v])
            break
        # try to find a good clique; pick a random start among high-degree uncovered vertices
        cand = set()
        for (u, v) in uncovered:
            cand.add(u)
            cand.add(v)
        deg = {x: 0 for x in cand}
        for (u, v) in uncovered:
            deg[u] += 1
            deg[v] += 1
        # choose start: random among top few by degree for diversity
        top = sorted(cand, key=lambda x: (-deg[x], x))
        start = top[0] if top else None
        clique = _max_clique_greedy(adj, n, uncovered, start)
        if clique is None:
            # fallback: take a single uncovered edge as a 2-clique
            e = next(iter(uncovered))
            clique = [e[0], e[1]]
        # remove all edges of this clique that are uncovered
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                uncovered.discard(e)
        cliques.append(clique)
    return cliques

def construct():
    deadline = time.time() + 100.0  # safety margin under 120s
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        adj, edges = _gen_graph(n, p, seed)
        cliques = _partition_instance(adj, edges, deadline)
        results.append(cliques)
    return results

# EVOLVE-BLOCK-END
