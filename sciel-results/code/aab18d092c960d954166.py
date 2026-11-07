"""Clique partition of split graphs via neighborhood-signature grouping + max-clique refinement."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import defaultdict
    from itertools import combinations

    _T0 = time.time()
    _DEADLINE = 150.0 - 15.0  # safety margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _is_clique(adj, verts):
        vs = list(verts)
        for i in range(len(vs)):
            a = vs[i]
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[a]:
                    return False
        return True

    def _max_clique_bk(adj, cand):
        """Bron-Kerbosch with pivot, returns maximum clique."""
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            U = P | X
            if U:
                piv = max(U, key=lambda v: len(adj[v] & P))
                cand_list = [v for v in P if v not in adj[piv]]
            else:
                cand_list = list(P)
            for v in cand_list:
                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P.discard(v)
                X.add(v)
        bk([], set(cand), set())
        return best

    def _solve_one(n, k, p, seed):
        adj = {v: set() for v in range(n)}
        for (u, v) in instance(n, k, p, seed):
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # Build cross adjacency: for each I-vertex, which C-vertices it connects to
        cross_to_c = {v: frozenset(adj[v] & set(C)) for v in I}

        # All edges
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        cliques = []
        covered = set()

        def add_clique(verts):
            nonlocal covered
            vs = sorted(set(verts))
            if len(vs) < 2:
                return
            # Verify clique
            if not _is_clique(adj, vs):
                return
            new_edges = []
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (vs[i], vs[j])
                    if e not in covered:
                        new_edges.append(e)
            if not new_edges:
                return
            for e in new_edges:
                covered.add(e)
            cliques.append(vs)

        # Phase 1: Cover C-C edges with full clique
        add_clique(C)

        # Phase 2: Group I-vertices by their C-neighborhood
        # For each group of I-vertices with the same C-neighborhood S,
        # we can form cliques: for each I-vertex v in group, clique = {v} ∪ (S ∩ uncovered)
        # But better: group I-vertices by C-neighborhood, and for each group,
        # the clique {v1, v2, ..., vm} ∪ S is NOT a clique unless vi-vj edges exist (they don't, I is independent)
        # So each I-vertex must be in separate cliques with subsets of C.
        # Optimal: for each I-vertex v, clique = {v} ∪ N_C(v) where N_C(v) = adj[v] ∩ C
        # But we want to minimize cliques, so we should share C-vertices across multiple I-cliques? No -
        # each edge must be in exactly one clique. So edge (c, v) can only be in one clique.
        # Therefore each cross edge (c,v) must be assigned to exactly one clique.
        # A clique containing v and some C-subset S' covers edges (c,v) for c in S'.
        # To minimize cliques, we want each clique to cover as many cross edges as possible.
        # Since I is independent, a clique can contain at most one I-vertex.
        # So each I-vertex v needs one clique: {v} ∪ S_v where S_v ⊆ N_C(v).
        # To minimize total cliques, we use one clique per I-vertex with S_v = N_C(v).
        # But wait - if two I-vertices v1, v2 share the same N_C, we still need separate cliques
        # because v1-v2 is not an edge. So minimum is |I| cliques for cross edges? No -
        # we can have a clique {v1, c1, c2} and another {v2, c3, c4} etc.
        # Actually each I-vertex needs its own clique since I is independent.
        # So cross edges require at least |I| cliques (one per I-vertex).
        # But we can choose which C-vertices go with which I-vertex to avoid conflicts.
        # Since each edge (c,v) must be in exactly one clique, and each I-vertex is in exactly one clique,
        # the clique for v_i is {v_i} ∪ S_i where S_i = N_C(v_i) exactly.
        # This gives |I| cliques for cross edges.

        # But we can do better: if some I-vertex has empty N_C, it has no cross edges, skip it.
        # Actually we MUST cover all cross edges. Each cross edge (c,v) is in the clique for v.
        # So the clique for v is {v} ∪ N_C(v). This is valid if N_C(v) is a clique in C (it is, C is a clique).
        # And {v} ∪ N_C(v) is a clique iff v is connected to all of N_C(v) (true by definition).

        # So Phase 2: for each I-vertex v with N_C(v) non-empty, add clique {v} ∪ N_C(v)
        for v in I:
            Nc = adj[v] & set(C)
            if Nc:
                add_clique([v] + list(Nc))

        # Phase 3: Cover any remaining edges (should be none for cross, but check C-C if Phase 1 failed)
        residual = [e for e in all_edges if e not in covered]
        if residual:
            # Build residual adjacency
            res_adj = {v: set() for v in range(n)}
            for (u, v) in residual:
                res_adj[u].add(v)
                res_adj[v].add(u)
            # Greedy max clique
            while residual and time.time() < _DEADLINE:
                start = max(range(n), key=lambda v: len(res_adj[v]))
                if len(res_adj[start]) < 1:
                    for (u, v) in residual:
                        if (u, v) not in covered:
                            add_clique([u, v])
                    break
                cand = {start} | res_adj[start]
                mc = _max_clique_bk(res_adj, cand)
                if len(mc) < 2:
                    for (u, v) in residual:
                        if (u, v) not in covered:
                            add_clique([u, v])
                    break
                before = len(covered)
                add_clique(mc)
                if len(covered) == before:
                    for (u, v) in residual:
                        if (u, v) not in covered:
                            add_clique([u, v])
                    break
                # Update residual
                new_residual = []
                for e in residual:
                    if e in covered:
                        continue
                    if e[0] in mc or e[1] in mc:
                        continue
                    new_residual.append(e)
                residual = new_residual
                for v in mc:
                    res_adj[v].clear()
                for (u, v) in residual:
                    if v in res_adj[u]:
                        res_adj[u].add(v)
                        res_adj[v].add(u)
            # Safety
            for (u, v) in all_edges:
                if (u, v) not in covered:
                    add_clique([u, v])

        # Phase 4: Local refinement - merge cliques if union is a clique and edge sets match
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) < 2:
                        continue
                    if not _is_clique(adj, uni):
                        continue
                    # Check edge sets
                    def edge_set(cl):
                        s = set()
                        for a in range(len(cl)):
                            for b in range(a + 1, len(cl)):
                                s.add((cl[a], cl[b]))
                        return s
                    ei = edge_set(cliques[i])
                    ej = edge_set(cliques[j])
                    eu = edge_set(sorted(uni))
                    if ei | ej == eu:
                        cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                        cliques.append(sorted(uni))
                        improved = True
                        break
                if improved:
                    break

        # Final validation
        for cl in cliques:
            assert len(cl) >= 2, f"Clique too small: {cl}"
            assert len(set(cl)) == len(cl), f"Duplicate vertices in clique: {cl}"
            assert _is_clique(adj, cl), f"Not a clique: {cl}"
        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (cl[a], cl[b])
                    assert e not in seen, f"Edge {e} in two cliques"
                    seen.add(e)
        assert seen == all_edges, "Not all edges covered"

        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
