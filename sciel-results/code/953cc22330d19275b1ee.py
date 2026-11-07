# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import defaultdict

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

    def _edge_count(clique, uncovered):
        """Count how many edges in this clique are still uncovered."""
        cnt = 0
        vs = clique
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                e = (min(vs[i], vs[j]), max(vs[i], vs[j]))
                if e in uncovered:
                    cnt += 1
        return cnt

    def _solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)

        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        C = set(range(k))
        I = list(range(k, n))

        # Precompute cross neighborhoods
        N_C = {v: adj[v] & C for v in I}
        N_I = {u: adj[u] & set(I) for u in C}

        uncovered = set(edge_set)
        cliques = []

        def add_clique(verts):
            vs = sorted(verts)
            if len(vs) < 2:
                return 0
            if not _is_clique(adj, vs):
                return 0
            cnt = 0
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (min(vs[i], vs[j]), max(vs[i], vs[j]))
                    if e in uncovered:
                        uncovered.discard(e)
                        cnt += 1
            if cnt > 0:
                cliques.append(vs)
            return cnt

        # Phase 1: Add the full clique C (covers all internal C-C edges)
        if k >= 2:
            add_clique(list(C))

        # Phase 2: Greedy selection with Lagrangian edge-weight guidance
        # We maintain weights on uncovered edges. Initially uniform.
        # After each selection, we boost weights on edges that are "hard" to cover
        # (incident to vertices with few remaining options).
        edge_weights = {}
        for e in uncovered:
            edge_weights[e] = 1.0

        # Precompute for each v in I: the set of C-neighbors
        # A candidate clique for v is {v} ∪ S where S ⊆ N_C[v]
        # The "value" is the weighted sum of uncovered edges covered.

        while uncovered and time.time() < _DEADLINE:
            best_clique = None
            best_score = -1

            # Candidate 1: Full C clique (if any C-C edges remain)
            c_edges_remaining = [(u, v) for u in C for v in C if u < v and (u, v) in uncovered]
            if c_edges_remaining:
                c_list = sorted(C)
                score = sum(edge_weights.get((min(a,b), max(a,b)), 0.0) for a in c_list for b in c_list if a < b and (min(a,b), max(a,b)) in uncovered)
                if score > best_score:
                    best_score = score
                    best_clique = c_list

            # Candidate 2: For each v in I, try {v} ∪ N_C[v] (full neighborhood)
            for v in I:
                Nv = N_C[v]
                if not Nv:
                    continue
                clique = [v] + sorted(Nv)
                # Compute weighted score
                score = 0.0
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (min(a, b), max(a, b))
                        if e in uncovered:
                            score += edge_weights.get(e, 0.0)
                if score > best_score:
                    best_score = score
                    best_clique = clique

            # Candidate 3: For each v in I, try {v} ∪ (subset of N_C[v] that maximizes
            # weighted uncovered edges). Since all edges in {v}∪S are cross edges (v to S)
            # plus C-C edges (already covered), the value is just the weighted sum of
            # (u,v) for u in S that are uncovered.
            # Actually {v} ∪ S only covers edges (u,v) for u in S, since I is independent.
            # Wait - it also covers edges within S, but those are C-C edges already covered.
            # So the new edges covered are exactly {(u,v) : u in S, (u,v) in uncovered}.
            # The best S for v is all u in N_C[v] where (u,v) is uncovered.
            for v in I:
                Nv = N_C[v]
                if not Nv:
                    continue
                # S = all u in N_C[v] such that (u,v) is uncovered
                S = [u for u in Nv if (min(u, v), max(u, v)) in uncovered]
                if not S:
                    continue
                clique = [v] + sorted(S)
                score = sum(edge_weights.get((min(u, v), max(u, v)), 0.0) for u in S)
                if score > best_score:
                    best_score = score
                    best_clique = clique

            if best_clique is None or best_score <= 0:
                # Fallback: pick any uncovered edge
                e = next(iter(uncovered))
                best_clique = sorted([e[0], e[1]])

            cnt = add_clique(best_clique)
            if cnt == 0:
                # Shouldn't happen, but safety
                break

            # Update edge weights: boost edges incident to vertices with few remaining
            # uncovered edges (they're "hard" to cover, so prioritize them)
            # Count remaining uncovered degree for each vertex
            rem_deg = defaultdict(int)
            for e in uncovered:
                rem_deg[e[0]] += 1
                rem_deg[e[1]] += 1

            # Boost weights for edges where both endpoints have low remaining degree
            for e in list(edge_weights.keys()):
                if e not in uncovered:
                    del edge_weights[e]
                    continue
                u, v = e
                du = rem_deg.get(u, 0)
                dv = rem_deg.get(v, 0)
                # Edges with lower remaining degree get higher weight
                boost = 1.0 + 2.0 / (1.0 + min(du, dv))
                edge_weights[e] = edge_weights[e] * boost

        # Cover any remaining edges with 2-cliques
        for e in sorted(uncovered):
            cliques.append(sorted([e[0], e[1]]))
            uncovered.discard(e)

        # Phase 3: Local refinement - try to merge cliques or replace pairs with better solutions
        improved = True
        while improved and time.time() < _DEADLINE - 5:
            improved = False

            # Try merging two cliques if their union is a clique and covers the same edges
            for i in range(len(cliques)):
                if improved:
                    break
                for j in range(i + 1, len(cliques)):
                    uni = set(cliques[i]) | set(cliques[j])
                    if len(uni) < 2:
                        continue
                    if not _is_clique(adj, uni):
                        continue
                    # Check that union's edge set equals the union of the two cliques' edge sets
                    def edge_set(cl):
                        s = set()
                        for a in range(len(cl)):
                            for b in range(a + 1, len(cl)):
                                s.add((min(cl[a], cl[b]), max(cl[a], cl[b])))
                        return s
                    ei = edge_set(cliques[i])
                    ej = edge_set(cliques[j])
                    eu = edge_set(sorted(uni))
                    if ei | ej == eu:
                        cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                        cliques.append(sorted(uni))
                        improved = True
                        break

            # Try replacing two cliques with a single larger clique
            # For each pair, check if there's a superset clique that covers all their edges
            if not improved:
                for i in range(len(cliques)):
                    if improved:
                        break
                    for j in range(i + 1, len(cliques)):
                        # All edges covered by cliques[i] and cliques[j]
                        all_edges_ij = set()
                        for cl in (cliques[i], cliques[j]):
                            for a in range(len(cl)):
                                for b in range(a + 1, len(cl)):
                                    all_edges_ij.add((min(cl[a], cl[b]), max(cl[a], cl[b])))
                        # Try to find a clique that covers all these edges
                        # The vertices involved
                        verts_involved = set(cliques[i]) | set(cliques[j])
                        # A clique covering all edges must contain all endpoints
                        # But it might need more vertices
                        # For now, just try the union of vertices
                        candidate = sorted(verts_involved)
                        if _is_clique(adj, candidate):
                            # Check it covers all edges
                            cand_edges = set()
                            for a in range(len(candidate)):
                                for b in range(a + 1, len(candidate)):
                                    cand_edges.add((min(candidate[a], candidate[b]), max(candidate[a], candidate[b])))
                            if all_edges_ij <= cand_edges:
                                cliques = [c for idx, c in enumerate(cliques) if idx not in (i, j)]
                                cliques.append(candidate)
                                improved = True
                                break

        # Final validation
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.add((u, v))

        for cl in cliques:
            assert len(cl) >= 2, f"Clique too small: {cl}"
            assert len(set(cl)) == len(cl), f"Duplicate vertices in clique: {cl}"
            assert _is_clique(adj, cl), f"Not a clique: {cl}"

        seen = set()
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (min(cl[a], cl[b]), max(cl[a], cl[b]))
                    assert e not in seen, f"Edge {e} in two cliques"
                    seen.add(e)
        assert seen == all_edges, f"Missing edges: {all_edges - seen}"

        return [list(map(int, cl)) for cl in cliques]

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _DEADLINE:
            # Fallback: one clique per edge
            out.append([list(e) for e in instance(*spec)])
            continue
        out.append(_solve_one(*spec))

    return out
# EVOLVE-BLOCK-END
