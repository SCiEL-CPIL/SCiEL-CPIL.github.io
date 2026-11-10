"""Edge clique partition via greedy set-cover with bitmask clique expansion."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 120.0  # wall-clock seconds; evaluator kills at 150s, keep margin


def _generate_graph(n, p, seed):
    """Return list of edges (u, v) with u < v using the same RNG as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _solve_instance(n, edges, time_budget):
    """
    Partition edges into cliques (each a list of >=2 vertices, a clique,
    covering every edge exactly once), minimizing the number of cliques.
    Returns list of cliques (vertex lists).
    """
    start = time.time()
    m = len(edges)
    if m == 0:
        return []

    # adjacency as bitmasks
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    # map each edge to an index
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # uncovered edges as a set of edge indices
    uncovered = set(range(m))

    # helper: is a set of vertices (bitmask) a clique?
    def is_clique(mask, verts):
        # verts is a list of vertices in the mask
        for i in range(len(verts)):
            vi = verts[i]
            # all other vertices in mask must be adjacent to vi
            others = mask & ~(1 << vi)
            if others & ~adj[vi]:
                return False
        return True

    # Greedy: build candidate cliques by expanding edges, pick best coverage.
    cliques = []

    # Precompute, for each vertex, its neighbor mask for fast expansion.
    # We'll use a greedy maximal-clique expansion from a seed edge.

    def expand_to_clique(seed_u, seed_v, allowed_mask):
        """
        Grow a clique starting from {seed_u, seed_v} within allowed_mask
        (vertices that are all mutually adjacent to current clique).
        Greedy: repeatedly add the vertex in the candidate pool with the
        highest degree to the pool (to keep the pool dense).
        Returns the clique as a list of vertices.
        """
        clique_mask = (1 << seed_u) | (1 << seed_v)
        # candidate pool: vertices in allowed_mask adjacent to both seed_u and seed_v
        pool = allowed_mask & adj[seed_u] & adj[seed_v]
        # exclude the two seeds from pool
        pool &= ~clique_mask
        while pool:
            # pick vertex in pool with max (popcount of pool & adj[v])
            best_v = -1
            best_score = -1
            p = pool
            while p:
                lsb = p & (-p)
                v = lsb.bit_length() - 1
                score = (pool & adj[v]).bit_count()
                if score > best_score:
                    best_score = score
                    best_v = v
                p &= p - 1
            if best_v < 0:
                break
            clique_mask |= (1 << best_v)
            pool &= adj[best_v]
            pool &= ~(1 << best_v)
        return clique_mask

    # We'll iterate: for each uncovered edge, try to expand it into a clique
    # restricted to vertices that are endpoints of uncovered edges (to maximize
    # coverage of uncovered edges). Pick the clique covering the most uncovered
    # edges.

    # To keep it efficient, we maintain a set of "active" vertices = endpoints of
    # uncovered edges.
    active_mask = 0
    for i in uncovered:
        u, v = edges[i]
        active_mask |= (1 << u) | (1 << v)

    # Greedy set-cover loop
    while uncovered:
        if time.time() - start > time_budget:
            break

        # For speed, sample a subset of uncovered edges if there are many.
        edge_list = list(uncovered)
        if len(edge_list) > 400:
            # sample
            sample = random.Random(12345).sample(edge_list, 400)
        else:
            sample = edge_list

        best_clique_mask = None
        best_cover = -1
        best_size = -1

        for ei in sample:
            u, v = edges[ei]
            # expand from this edge within active vertices
            cand = expand_to_clique(u, v, active_mask)
            # count how many uncovered edges this clique covers
            cover = 0
            verts = [x for x in range(n) if (cand >> x) & 1]
            for a in range(len(verts)):
                va = verts[a]
                for b in range(a + 1, len(verts)):
                    vb = verts[b]
                    key = (va, vb)
                    idx = edge_index.get(key)
                    if idx is not None and idx in uncovered:
                        cover += 1
            # prefer more uncovered edges covered; tie-break by larger clique
            if cover > best_cover or (cover == best_cover and len(verts) > best_size):
                best_cover = cover
                best_size = len(verts)
                best_clique_mask = cand

        if best_clique_mask is None or best_cover <= 0:
            # fallback: take a single uncovered edge as a 2-vertex clique
            # pick the first uncovered edge
            ei = next(iter(uncovered))
            u, v = edges[ei]
            best_clique_mask = (1 << u) | (1 << v)
            best_cover = 1

        # commit this clique
        verts = [x for x in range(n) if (best_clique_mask >> x) & 1]
        cliques.append(verts)
        # remove covered edges from uncovered
        for a in range(len(verts)):
            va = verts[a]
            for b in range(a + 1, len(verts)):
                vb = verts[b]
                key = (va, vb)
                idx = edge_index.get(key)
                if idx is not None and idx in uncovered:
                    uncovered.discard(idx)

        # update active_mask
        active_mask = 0
        for i in uncovered:
            u, v = edges[i]
            active_mask |= (1 << u) | (1 << v)

    # If we broke out early (deadline), cover remaining edges as 2-vertex cliques
    if uncovered:
        for ei in uncovered:
            u, v = edges[ei]
            cliques.append([u, v])
        uncovered.clear()

    return cliques


def construct():
    """Return two lists of cliques for the two fixed instances."""
    t0 = time.time()
    total_budget = _DEADLINE - 2.0  # safety margin

    results = []
    params = ((32, 0.6, 601), (64, 0.55, 602))

    # Allocate time: give the larger instance more budget
    # First instance: 30% of budget, second: 70%
    budgets = []
    remaining = total_budget
    for i, (n, p, seed) in enumerate(params):
        if i == len(params) - 1:
            budgets.append(remaining)
        else:
            b = remaining * 0.3
            budgets.append(b)
            remaining -= b

    for (n, p, seed), tb in zip(params, budgets):
        edges = _generate_graph(n, p, seed)
        cliques = _solve_instance(n, edges, tb)
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
