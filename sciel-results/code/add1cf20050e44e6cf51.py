def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = start_time + 130.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, ordering='ascending'):
        cross_edges = instance(n, k, p, seed)

        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)

        # Build C-C edge structure
        # C-vertices are 0..k-1, all connected to each other (clique)
        # We track which C-C edges are still "free" (not used in any mixed clique)
        # A C-C edge (a,b) is used when we form a mixed clique {v, a, b}
        # where v is an I-vertex adjacent to both a and b

        # Represent free C-C edges as a set of frozensets or just track via a 2D structure
        # For efficiency with k up to 30, use a list of sets
        free_edges = [set() for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                free_edges[i].add(j)
                free_edges[j].add(i)

        # Count of free edges
        free_count = k * (k - 1) // 2

        def use_edge(a, b):
            """Mark C-C edge (a,b) as used"""
            nonlocal free_count
            if b in free_edges[a]:
                free_edges[a].discard(b)
                free_edges[b].discard(a)
                free_count -= 1

        mixed_cliques = []

        # Determine order of I vertices
        I_vertices = list(range(k, n))
        if ordering == 'ascending':
            I_vertices.sort(key=lambda x: len(N[x]))
        elif ordering == 'descending':
            I_vertices.sort(key=lambda x: -len(N[x]))
        elif ordering == 'random':
            Random(seed + 999).shuffle(I_vertices)
        elif ordering == 'weighted_asc':
            # Sort by degree, breaking ties randomly
            rng_tie = Random(seed + 777)
            I_vertices.sort(key=lambda x: (len(N[x]), rng_tie.random()))
        elif ordering == 'weighted_desc':
            rng_tie = Random(seed + 888)
            I_vertices.sort(key=lambda x: (-len(N[x]), rng_tie.random()))

        for v in I_vertices:
            neighbors = N[v]
            if not neighbors:
                continue

            # Find all free C-C edges where both endpoints are neighbors of v
            # These are the edges we can "absorb" into a mixed clique with v
            # We want to select a matching (set of disjoint edges) among these
            # to maximize the number of C-C edges covered

            # Build list of available edges (both endpoints in neighbors and edge is free)
            available = []
            ns = set(neighbors)
            for c_a in ns:
                for c_b in free_edges[c_a]:
                    if c_b > c_a and c_b in ns:
                        available.append((c_a, c_b))

            if not available:
                # Can only form pairs (v, c) for each neighbor
                for c in neighbors:
                    mixed_cliques.append([int(v), int(c)])
                continue

            # Greedy matching: pick edges that are "most valuable"
            # Value of an edge = 1 (it covers one C-C edge)
            # But we also want to leave good edges for other I-vertices
            # Strategy: pick edges where the endpoints have low "alternative" (low degree in the subgraph)
            # Or: pick edges that are "rare" (endpoints have few other free edges to neighbors of v)

            # For simplicity, use a greedy matching that prioritizes edges
            # whose endpoints have fewer remaining free edges overall (more "committed")
            # Actually, let's think about it differently:
            # We want to maximize the number of C-C edges we cover with v.
            # A maximum matching in the subgraph induced by neighbors on free edges.
            # Greedy: sort edges by (deg_in_subgraph[a] + deg_in_subgraph[b]) ascending,
            # so we pick "leaf-like" edges first, which is a good heuristic for maximum matching.

            # Compute degree of each neighbor in the available subgraph
            sub_deg = {c: 0 for c in ns}
            for a, b in available:
                sub_deg[a] += 1
                sub_deg[b] += 1

            # Sort by sum of subgraph degrees (ascending - pick low-degree edges first)
            available.sort(key=lambda e: sub_deg[e[0]] + sub_deg[e[1]])

            matched = set()
            pairings = []
            for c_a, c_b in available:
                if c_a in matched or c_b in matched:
                    continue
                # Verify edge is still free
                if c_b not in free_edges[c_a]:
                    continue
                matched.add(c_a)
                matched.add(c_b)
                pairings.append((c_a, c_b))
                use_edge(c_a, c_b)

            for c_a, c_b in pairings:
                mixed_cliques.append([int(v), int(c_a), int(c_b)])

            for c in neighbors:
                if c not in matched:
                    mixed_cliques.append([int(v), int(c)])

        # Cover remaining C-C edges with pure-C cliques
        pure_cliques = []

        # Collect all remaining free edges
        remaining_edges = []
        for i in range(k):
            for j in free_edges[i]:
                if j > i:
                    remaining_edges.append((i, j))

        # Greedy maximal clique extraction
        while remaining_edges:
            if time.time() > deadline - 2.0:
                for a, b in remaining_edges:
                    pure_cliques.append([int(a), int(b)])
                break

            # Pick a seed edge
            # Strategy: pick edge with max combined degree in remaining graph
            rem_set = set(remaining_edges)
            deg_rem = {}
            for a, b in remaining_edges:
                deg_rem[a] = deg_rem.get(a, 0) + 1
                deg_rem[b] = deg_rem.get(b, 0) + 1

            best_edge = max(remaining_edges, key=lambda e: deg_rem.get(e[0], 0) + deg_rem.get(e[1], 0))
            seed_v = best_edge[0]

            # Build maximal clique containing seed_v in the remaining graph
            S = {seed_v}
            # Candidates: vertices adjacent to seed_v in remaining graph
            cand = set()
            for a, b in remaining_edges:
                if a == seed_v:
                    cand.add(b)
                elif b == seed_v:
                    cand.add(a)

            changed = True
            while changed:
                changed = False
                # Find vertices in cand that are connected to all in S
                for w in list(cand):
                    ok = True
                    for s in S:
                        if w == s:
                            continue
                        edge = (min(w, s), max(w, s))
                        if edge not in rem_set:
                            ok = False
                            break
                    if ok:
                        S.add(w)
                        cand.discard(w)
                        # Remove w from other candidates that aren't connected to w
                        new_cand = set()
                        for x in cand:
                            edge_xw = (min(x, w), max(x, w))
                            if edge_xw in rem_set:
                                new_cand.add(x)
                        cand = new_cand
                        changed = True
                        break

            if len(S) >= 2:
                pure_cliques.append([int(x) for x in sorted(S)])
                S_set = S
                remaining_edges = [(a, b) for a, b in remaining_edges
                                   if not (a in S_set and b in S_set)]
            else:
                # Just take the seed edge
                a, b = best_edge
                pure_cliques.append([int(a), int(b)])
                remaining_edges.remove(best_edge)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    orderings = ['ascending', 'descending', 'weighted_asc', 'weighted_desc', 'random']

    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')

        if time.time() > deadline - 20.0:
            ce = instance(n, k, p, seed)
            cliques = [[int(u) for u in range(k)]]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue

        # Try multiple orderings
        for ordering in orderings:
            if time.time() > deadline - 25.0:
                break
            cliques = solve_instance(n, k, p, seed, ordering)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        # Try a few random orderings
        for i in range(5):
            if time.time() > deadline - 25.0:
                break
            cliques = solve_instance(n, k, p, seed, 'random')
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
