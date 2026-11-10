"""Clique partition of split graphs exploiting the complete-clique structure."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _max_clique_bfs(adj, n):
        """Bron–Kerbosch with pivot + degeneracy ordering (BFS degeneracy)."""
        # Build neighbor lists
        nbr = [sorted(adj[i]) for i in range(n)]
        # Compute degeneracy ordering
        order = list(range(n))
        pos = {v: i for i, v in enumerate(order)}
        deg = [len(nbr[v]) for v in range(n)]
        # Simple degeneracy: repeatedly remove min-degree vertex
        remaining = set(range(n))
        d_order = []
        local_deg = dict(deg)
        while remaining:
            # find min degree
            min_v = min(remaining, key=lambda v: local_deg[v])
            d_order.append(min_v)
            remaining.discard(min_v)
            for u in nbr[min_v]:
                if u in remaining:
                    local_deg[u] -= 1
        # Reverse for processing (highest degeneracy first)
        order = list(reversed(d_order))
        pos = {v: i for i, v in enumerate(order)}

        best = []

        def bronk(P, X, R):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if len(P) + len(X) <= len(best):
                return
            # Pivot selection
            if P:
                pivot = max(P, key=lambda v: len(nbr[v]) - len(set(nbr[v]) & X))
                candidates = [v for v in P if v not in nbr[pivot]]
            else:
                candidates = list(P)
            for v in candidates:
                R.append(v)
                P2 = [u for u in P if u in nbr[v]]
                X2 = [u for u in X if u in nbr[v]]
                bronk(P2, X2, R)
                R.pop()
                P.remove(v)
                X.append(v)

        # Initial call with all vertices in P, in degeneracy order
        P = list(order)
        X = []
        R = []
        bronk(P, X, R)
        return best

    def _greedy_clique_partition(adj, n):
        """Greedy: repeatedly extract a max clique, remove its edges."""
        # Work on a mutable adjacency (sets)
        adj_sets = [set(adj[i]) for i in range(n)]
        cliques = []
        remaining = set(range(n))
        while remaining:
            # Build subgraph on remaining
            sub_n = len(remaining)
            if sub_n <= 1:
                break
            # Find max clique in subgraph induced by remaining
            # Map remaining to 0..sub_n-1
            idx = {v: i for i, v in enumerate(remaining)}
            sub_adj = [set() for _ in range(sub_n)]
            for v in remaining:
                vi = idx[v]
                for u in adj_sets[v]:
                    if u in remaining and u > v:
                        ui = idx[u]
                        sub_adj[vi].add(ui)
                        sub_adj[ui].add(vi)
            mc = _max_clique_bfs(sub_adj, sub_n)
            if len(mc) < 2:
                # Shouldn't happen if remaining has edges; fallback
                # Take any edge
                found = False
                for v in remaining:
                    for u in adj_sets[v]:
                        if u in remaining and u > v:
                            cliques.append([v, u])
                            for x in [v, u]:
                                adj_sets[x].discard(v if x == u else u)
                            remaining.discard(v)
                            remaining.discard(u)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
                continue
            clique = [remaining[i] for i in mc]
            cliques.append(clique)
            # Remove edges of clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    adj_sets[a].discard(b)
                    adj_sets[b].discard(a)
            # Remove vertices with no remaining edges
            to_remove = set()
            for v in remaining:
                if not any(u in remaining for u in adj_sets[v]):
                    to_remove.add(v)
            remaining -= to_remove
        return cliques

    def partition_split_graph(n, k, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = []

        # Part 1: Partition the k-clique (vertices 0..k-1)
        # The clique is complete, so we can greedily extract large cliques.
        # But we must be careful: some edges from clique vertices go to independent
        # set vertices. We'll handle those separately.
        # Strategy: First, partition the k-clique into cliques.
        # Then, for each independent-set vertex, its edges to the clique must be
        # covered. Each such edge (c, w) where c < k, w >= k must be in a clique
        # containing both c and w. Since w has no edges to other independent-set
        # vertices, any clique containing w can only contain w and clique vertices
        # that are all mutually adjacent (which they are, since clique is complete)
        # and adjacent to w.

        # So: For each independent vertex w, let N(w) = neighbors of w in the clique.
        # The edges from w to N(w) must be covered by cliques each containing w
        # and a subset of N(w) that forms a clique (always true since N(w) subset of
        # clique which is complete). But those clique vertices also have edges among
        # themselves that need coverage.

        # Better approach: 
        # 1. For each independent vertex w, create a clique {w} ∪ S_w where S_w is
        #    chosen to cover w's edges. Since all S_w vertices are in the clique,
        #    any subset works. To minimize total cliques, we want large S_w.
        #    But S_w vertices' mutual edges will be covered by the clique partition.
        # 2. The remaining edges in the k-clique (those not covered by being in
        #    some S_w clique) need a clique partition of the k-clique.

        # Actually, every edge in the k-clique must be in exactly one clique.
        # Edges (c1, c2) where c1, c2 < k can be covered either by:
        #   - A clique in the k-clique partition, OR
        #   - A clique {w} ∪ S_w where w >= k and {c1, c2} ⊆ S_w
        # But if (c1,c2) is in {w}∪S_w, it's covered there. It can't also be in
        # the k-clique partition. So we need to coordinate.

        # Simpler: Just do a greedy max-clique extraction on the whole graph.
        # The structure will naturally find cliques like {w} ∪ N(w) and large
        # clique subsets.

        return _greedy_clique_partition(adj, n)

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            # Emergency fallback: one edge per clique
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue

        edges = instance(*spec)
        n, k = spec[0], spec[1]
        cliques = partition_split_graph(n, k, edges)

        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])

        results.append(final_cliques)

    return results
# EVOLVE-BLOCK-END
