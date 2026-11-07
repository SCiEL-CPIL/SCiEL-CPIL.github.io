def construct():
    import numpy as np
    from collections import defaultdict

    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        return cross

    def solve_clique_partition(n, k, p, seed):
        cross = instance(n, k, p, seed)
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        # Clique on first k
        for i in range(k):
            for j in range(i + 1, k):
                adj[i][j] = True
                adj[j][i] = True
        # Cross edges
        for (u, v) in cross:
            adj[u][v] = True
            adj[v][u] = True

        # Find all maximal cliques
        # Bron-Kerbosch with pivot
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(tuple(sorted(R)))
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(P & adj_set[v]) + len(X & adj_set[v]))
                candidates = P - adj_set[pivot]
            else:
                candidates = P
            for v in list(candidates):
                bronk(R | {v}, P & adj_set[v], X & adj_set[v])
                P.remove(v)
                X.add(v)

        adj_set = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_set[i].add(j)

        maximal_cliques = []
        bronk(set(), set(range(n)), set())

        # Remove cliques of size < 2
        maximal_cliques = [c for c in maximal_cliques if len(c) >= 2]

        # Map cliques to bitmasks
        cliques = list(maximal_cliques)
        num_cliques = len(cliques)

        # Build edge list
        edges = []
        edge_to_cliques = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edge_id = len(edges)
                    edges.append((i, j))
                    for ci, clique in enumerate(cliques):
                        if i in clique and j in clique:
                            edge_to_cliques[edge_id].append(ci)

        num_edges = len(edges)

        # Greedy set cover with local search
        # Start with all edges uncovered
        covered = [False] * num_edges
        selected = set()

        # Greedy: repeatedly pick the clique that covers the most uncovered edges
        while any(not c for c in covered):
            best_clique = -1
            best_count = -1
            for ci in range(num_cliques):
                if ci in selected:
                    continue
                count = 0
                for ei, (i, j) in enumerate(edges):
                    if not covered[ei] and i in cliques[ci] and j in cliques[ci]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_clique = ci
            if best_clique == -1:
                break
            selected.add(best_clique)
            for ei, (i, j) in enumerate(edges):
                if i in cliques[best_clique] and j in cliques[best_clique]:
                    covered[ei] = True

        # Local search: try to remove a clique and re-cover
        improved = True
        while improved:
            improved = False
            for ci in list(selected):
                # Try removing ci
                selected.remove(ci)
                # Find uncovered edges
                uncovered = []
                for ei, (i, j) in enumerate(edges):
                    if i in cliques[ci] and j in cliques[ci]:
                        # Check if still covered by other selected cliques
                        still_covered = False
                        for cj in selected:
                            if i in cliques[cj] and j in cliques[cj]:
                                still_covered = True
                                break
                        if not still_covered:
                            uncovered.append(ei)
                if not uncovered:
                    improved = True
                    continue
                # Try to re-cover with one clique
                best_replacement = -1
                best_coverage = -1
                for cj in range(num_cliques):
                    if cj in selected:
                        continue
                    cov = 0
                    for ei in uncovered:
                        i, j = edges[ei]
                        if i in cliques[cj] and j in cliques[cj]:
                            cov += 1
                    if cov > best_coverage:
                        best_coverage = cov
                        best_replacement = cj
                if best_coverage == len(uncovered):
                    selected.add(best_replacement)
                    improved = True
                    continue
                # Try to re-cover with two cliques
                found_two = False
                for cj1 in range(num_cliques):
                    if cj1 in selected:
                        continue
                    cov1 = 0
                    for ei in uncovered:
                        i, j = edges[ei]
                        if i in cliques[cj1] and j in cliques[cj1]:
                            cov1 += 1
                    if cov1 < len(uncovered):
                        continue
                    remaining = []
                    for ei in uncovered:
                        i, j = edges[ei]
                        if not (i in cliques[cj1] and j in cliques[cj1]):
                            remaining.append(ei)
                    if not remaining:
                        continue
                    for cj2 in range(num_cliques):
                        if cj2 in selected or cj2 == cj1:
                            continue
                        cov2 = 0
                        for ei in remaining:
                            i, j = edges[ei]
                            if i in cliques[cj2] and j in cliques[cj2]:
                                cov2 += 1
                        if cov2 == len(remaining):
                            selected.add(cj1)
                            selected.add(cj2)
                            found_two = True
                            break
                    if found_two:
                        break
                if found_two:
                    improved = True
                    continue
                # Couldn't replace, put ci back
                selected.add(ci)

        # Convert to list of lists
        result = [list(cliques[ci]) for ci in sorted(selected)]
        return result

    result = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve_clique_partition(n, k, p, seed)
        result.append(cliques)
    return result
