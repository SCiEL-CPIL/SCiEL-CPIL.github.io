def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # Neighbor sets for independent set vertices
        neighbor_sets = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if (u, v) in edge_set]
            neighbor_sets[v] = set(neighbors)

        # For each independent set vertex v, compute the set of clique-edges it covers
        # E_v = { (u, w) : u, w in N(v), u < w }
        v_edge_sets = {}
        for v in range(k, n):
            S = neighbor_sets[v]
            S_list = sorted(S)
            es = set()
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    es.add((S_list[i], S_list[j]))
            v_edge_sets[v] = es

        # Sort independent vertices by number of clique-edges they cover (descending)
        indep_vertices = sorted(range(k, n), key=lambda v: len(v_edge_sets[v]), reverse=True)

        # Greedy: select a maximal set of independent vertices such that no two
        # share a clique-edge. This ensures the cliques [v] + N(v) are edge-disjoint.
        selected = []
        covered_clique_edges = set()
        for v in indep_vertices:
            es = v_edge_sets[v]
            if es and not (es & covered_clique_edges):
                selected.append(v)
                covered_clique_edges |= es

        # Build the partition
        selected_cliques = []

        # For each selected independent vertex v, add clique [v] + N(v)
        for v in selected:
            S = neighbor_sets[v]
            clique = sorted(list(S) + [v])
            selected_cliques.append(clique)

        # For each non-selected independent vertex v, we need to cover its edges to N(v)
        # These edges (u, v) for u in N(v) are not covered by any selected clique
        # (since selected cliques only cover clique-edges within C, not cross-edges).
        # Wait - actually, the cross-edges (u, v) are only in clique [v] + N(v) for v itself.
        # Since v is not selected, we need to cover edges (u, v) for u in N(v).
        # We can add 2-cliques [u, v] for each u in N(v).
        selected_set = set(selected)
        for v in range(k, n):
            if v not in selected_set:
                for u in sorted(neighbor_sets[v]):
                    selected_cliques.append([u, v])

        # Now cover remaining clique-edges (edges within C = [0, k))
        # These are edges (u, w) for u, w in [0, k), u < w, that are not in covered_clique_edges
        remaining_clique_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) in edge_set and (u, w) not in covered_clique_edges:
                    remaining_clique_edges.add((u, w))

        # Cover remaining clique-edges with cliques. Since they're all within C,
        # we can try to find maximal cliques in the remaining edge subgraph.
        # Simple approach: greedy clique cover on the remaining edges.
        # Build adjacency for remaining edges
        rem_adj = {u: set() for u in range(k)}
        for (u, w) in remaining_clique_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)

        # Greedy: find a maximal clique, remove its edges, repeat
        rem_edges = set(remaining_clique_edges)
        while rem_edges:
            # Pick a vertex with highest degree in remaining edges
            best_v = max(rem_edges, key=lambda x: len(rem_adj[x] & {y for (a, b) in rem_edges for y in [a, b] if x in (a, b)}))
            # Actually, let's just pick any vertex and build a greedy clique
            # Find a vertex that has at least one remaining edge
            v0 = None
            for (a, b) in rem_edges:
                v0 = a
                break
            if v0 is None:
                break
            
            # Greedily build a clique starting from v0
            clique = [v0]
            # Vertices that can be added: neighbors of all current clique members in rem_adj
            candidates = set(rem_adj[v0])
            # Only consider vertices connected via remaining edges
            for v in sorted(candidates):
                # Check if v is connected to all current clique members via remaining edges
                can_add = True
                for c in clique:
                    if v not in rem_adj[c]:
                        can_add = False
                        break
                if can_add:
                    clique.append(v)
            
            # Remove all edges within this clique from rem_edges
            clique_set = set(clique)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    rem_edges.discard(e)
            
            if len(clique) >= 2:
                selected_cliques.append(sorted(clique))

        return [list(c) for c in selected_cliques]

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [clique_partition(*spec) for spec in specs]
