"""Clique partition of split graphs via greedy set-cover with edge-packing."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build cross-edge adjacency
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        # All C-edges
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        # All edges that need covering
        all_edges = set(C_edges)
        for i in range(k, n):
            for u in N[i]:
                all_edges.add((min(u, i), max(u, i)))

        def greedy_set_cover(all_edges, time_budget):
            """Greedy set cover: at each step pick the candidate clique that
            covers the most uncovered edges, subject to no overlap with already-chosen cliques."""
            uncovered = set(all_edges)
            chosen = []

            # Precompute for each I-vertex, the C-edges within its neighborhood
            # A clique centered at I-vertex i with C-subset S covers:
            #   - cross edges (u, i) for u in S
            #   - C-edges (u, v) for u, v in S
            # A C-only clique with subset S covers C-edges within S.

            # Candidate generation: for each I-vertex, try various S subsets
            # For C-only, try various S subsets

            while uncovered:
                if time.time() > time_budget:
                    break

                best_gain = 0
                best_clique = None
                best_edge_set = None

                # Try I-centered cliques
                for i in range(k, n):
                    if not N[i]:
                        continue
                    # Find max subset S of N[i] such that:
                    # 1. All C-edges within S are uncovered
                    # 2. All cross edges (u,i) for u in S are uncovered
                    # We want to maximize |S| (each vertex in S covers 1 cross edge + C-edges)

                    # Greedy: build S by adding vertices that keep the C-subgraph a clique
                    # and whose cross edges are uncovered
                    S = set()
                    # Sort N[i] by how many uncovered C-edges they have
                    candidates = sorted(N[i], key=lambda u: sum(
                        1 for v in N[i] if (min(u, v), max(u, v)) in uncovered
                    ), reverse=True)

                    for u in candidates:
                        if u not in N[i]:
                            continue
                        # Check cross edge (u, i) is uncovered
                        cross_e = (min(u, i), max(u, i))
                        if cross_e not in uncovered:
                            continue
                        # Check all C-edges (u, v) for v in S are uncovered
                        can_add = True
                        for v in S:
                            ce = (min(u, v), max(u, v))
                            if ce not in uncovered:
                                can_add = False
                                break
                        if can_add:
                            S.add(u)

                    if len(S) >= 1:
                        # Compute gain
                        gain = len(S)  # cross edges
                        S_list = sorted(S)
                        for a in range(len(S_list)):
                            for b in range(a + 1, len(S_list)):
                                if (S_list[a], S_list[b]) in uncovered:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_clique = sorted([i] + S_list)
                            best_edge_set = set()
                            for u in S_list:
                                best_edge_set.add((min(u, i), max(u, i)))
                            for a in range(len(S_list)):
                                for b in range(a + 1, len(S_list)):
                                    best_edge_set.add((S_list[a], S_list[b]))

                # Try C-only cliques
                # Find remaining C-edges and try to form a large clique
                rem_C = set()
                for e in uncovered:
                    u, v = e
                    if u < k and v < k:
                        rem_C.add(e)

                if rem_C:
                    # Build adjacency for remaining C-edges
                    rem_adj = defaultdict(set)
                    for u, v in rem_C:
                        rem_adj[u].add(v)
                        rem_adj[v].add(u)

                    # Greedy: find a large clique in rem_C
                    # Try from each vertex
                    best_C_S = set()
                    for start in list(rem_adj.keys()):
                        S = {start}
                        candidates = sorted(rem_adj[start], key=lambda x: len(
                            [y for y in rem_adj[start] if (min(x, y), max(x, y)) in rem_C]
                        ), reverse=True)
                        for u in candidates:
                            if u not in rem_adj:
                                continue
                            can_add = True
                            for v in S:
                                if (min(u, v), max(u, v)) not in rem_C:
                                    can_add = False
                                    break
                            if can_add:
                                S.add(u)
                        if len(S) > len(best_C_S):
                            best_C_S = set(S)

                    if len(best_C_S) >= 2:
                        S_list = sorted(best_C_S)
                        gain = 0
                        for a in range(len(S_list)):
                            for b in range(a + 1, len(S_list)):
                                if (S_list[a], S_list[b]) in uncovered:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_clique = S_list
                            best_edge_set = set()
                            for a in range(len(S_list)):
                                for b in range(a + 1, len(S_list)):
                                    best_edge_set.add((S_list[a], S_list[b]))

                if best_clique is None:
                    # Fallback: just take any single edge
                    e = next(iter(uncovered))
                    best_clique = sorted([e[0], e[1]])
                    best_edge_set = {e}
                else:
                    if best_edge_set is None:
                        best_edge_set = set()
                        verts = sorted(best_clique)
                        for a in range(len(verts)):
                            for b in range(a + 1, len(verts)):
                                best_edge_set.add((verts[a], verts[b]))

                chosen.append(best_clique)
                uncovered -= best_edge_set

            # Verify all edges are covered; if not, add remaining as pairs
            if uncovered:
                for e in uncovered:
                    # Check if this edge is already in a chosen clique
                    covered = False
                    for cl in chosen:
                        if e[0] in cl and e[1] in cl:
                            covered = True
                            break
                    if not covered:
                        chosen.append(sorted([e[0], e[1]]))

            return chosen

        def local_improve(cliques, time_budget):
            """Try to reduce clique count by:
            1. Merging two cliques if their union is a valid clique
            2. Moving a vertex from one clique to another if valid
            """
            best = list(cliques)
            best_count = len(best)

            for iteration in range(200):
                if time.time() > time_budget:
                    break

                improved = False

                # Try merging pairs of cliques
                for i in range(len(best)):
                    if time.time() > time_budget:
                        break
                    for j in range(i + 1, len(best)):
                        merged = set(best[i]) | set(best[j])
                        if len(merged) < 2:
                            continue
                        merged_list = sorted(merged)
                        # Check all pairs are edges
                        valid = True
                        for a in range(len(merged_list)):
                            for b in range(a + 1, len(merged_list)):
                                e = (merged_list[a], merged_list[b])
                                if e not in all_edges:
                                    valid = False
                                    break
                                # Check not in any other clique
                                for other_idx, other in enumerate(best):
                                    if other_idx == i or other_idx == j:
                                        continue
                                    if merged_list[a] in other and merged_list[b] in other:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            new_best = [c for idx, c in enumerate(best) if idx != i and idx != j]
                            new_best.append(merged_list)
                            if len(new_best) < best_count:
                                best = new_best
                                best_count = len(best)
                                improved = True
                                break
                    if improved:
                        break

                if not improved:
                    # Try moving a vertex between cliques
                    for i in range(len(best)):
                        if time.time() > time_budget:
                            break
                        for v in best[i]:
                            for j in range(len(best)):
                                if i == j:
                                    continue
                                # Try moving v from clique i to clique j
                                new_i = [x for x in best[i] if x != v]
                                new_j = sorted(set(best[j]) | {v})
                                if len(new_i) < 2 or len(new_j) < 2:
                                    continue
                                # Validate
                                valid = True
                                for a in range(len(new_j)):
                                    for b in range(a + 1, len(new_j)):
                                        e = (new_j[a], new_j[b])
                                        if e not in all_edges:
                                            valid = False
                                            break
                                        for other_idx, other in enumerate(best):
                                            if other_idx == i or other_idx == j:
                                                continue
                                            if new_j[a] in other and new_j[b] in other:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if not valid:
                                        break
                                if valid and len(new_i) >= 2:
                                    new_best = list(best)
                                    new_best[i] = new_i
                                    new_best[j] = new_j
                                    if len(new_best) < best_count:
                                        best = new_best
                                        best_count = len(new_best)
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break

                if not improved:
                    break

            return best

        # Generate initial solution with greedy set cover
        time_per_instance = (deadline - start_time) / 3.0
        t_budget = start_time + time_per_instance

        cliques = greedy_set_cover(all_edges, t_budget)

        # Local improvement
        if time.time() < deadline - 5:
            cliques = local_improve(cliques, t_budget)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        # Ensure all are Python ints and lists
        results.append([[int(v) for v in clique] for clique in cliques])

    return results
# EVOLVE-BLOCK-END
