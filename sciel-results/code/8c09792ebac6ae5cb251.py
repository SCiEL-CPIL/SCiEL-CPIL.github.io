"""Rewritten program: minimum clique cover via exact set cover on maximal cliques."""
import sys
from itertools import combinations

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def get_maximal_cliques(n, edges):
        """Find all maximal cliques in a graph given n vertices and edge set."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        # Bron-Kerbosch with pivot to find all maximal cliques
        maximal_cliques = []

        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X with maximum degree in P
            if P or X:
                u = max(list(P) + list(X), key=lambda x: len(adj[x] & P))
                candidates = P - adj[u]
            else:
                candidates = set()

            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bronk(frozenset(), set(range(n)), set())
        return maximal_cliques

    def solve_min_clique_cover(n, edges):
        """Solve minimum clique cover exactly using branch and bound."""
        if not edges:
            return 0

        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        num_edges = len(edge_set)

        # Find maximal cliques
        max_cliques = get_maximal_cliques(n, list(edge_set))

        # For each maximal clique, compute the set of edges it covers
        clique_edges = []
        for clique in max_cliques:
            verts = list(clique)
            edges_in_clique = set()
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                    if e in edge_set:
                        edges_in_clique.add(e)
            if edges_in_clique:
                clique_edges.append(edges_in_clique)

        # If no cliques with edges, something is wrong
        if not clique_edges:
            return num_edges

        # Convert to bitmasks for efficiency
        # Map each edge to an index
        edge_list = sorted(edge_set)
        edge_index = {e: i for i, e in enumerate(edge_list)}

        # Represent each clique's edge set as a bitmask
        # Use Python integers as bitmasks (num_edges <= ~4000 for n=90, which is fine)
        num_cliques = len(clique_edges)
        clique_masks = []
        for ce in clique_edges:
            mask = 0
            for e in ce:
                mask |= (1 << edge_index[e])
            clique_masks.append(mask)

        all_edges_mask = (1 << num_edges) - 1

        # Branch and bound
        best = [num_edges]  # Upper bound: cover each edge individually (each edge is a clique of size 2)
        
        # Actually, each edge itself is a clique of size 2, so we can always cover with num_edges cliques.
        # But we want to minimize. Let's use a better initial upper bound.
        # Greedy: repeatedly pick the clique covering the most uncovered edges
        uncovered = all_edges_mask
        greedy_count = 0
        while uncovered:
            best_idx = -1
            best_covered = 0
            for i in range(num_cliques):
                covered = bin(clique_masks[i] & uncovered).count('1')
                if covered > best_covered:
                    best_covered = covered
                    best_idx = i
            if best_idx == -1 or best_covered == 0:
                # Should not happen if cliques cover all edges
                greedy_count += bin(uncovered).count('1')
                break
            uncovered &= ~clique_masks[best_idx]
            greedy_count += 1
        best[0] = greedy_count

        # Lower bound: fractional packing
        # For a lower bound, we can use: at least ceil(num_edges / max_clique_edge_count)
        # But a better bound: for each edge, it must be in at least one clique.
        # Simple bound: the maximum number of edges in any single clique gives a lower bound of ceil(num_edges / max_edges_in_clique)
        max_edges_in_clique = max(bin(m).count('1') for m in clique_masks)
        lb_simple = (num_edges + max_edges_in_clique - 1) // max_edges_in_clique

        # A stronger lower bound using LP relaxation is complex. Let's use a simpler combinatorial bound.
        # For each edge e, let d(e) be the number of cliques containing e.
        # We need at least 1 clique for each edge, but cliques can cover multiple edges.
        # The fractional cover number is at least the maximum over all subsets of edges of (|S| / max_{clique} |S ∩ clique|)
        # This is hard to compute exactly. Let's use a greedy lower bound.
        
        # Use branch and bound with the simple lower bound
        import sys
        sys.setrecursionlimit(10000)

        # Sort cliques by size (larger first) for better pruning
        clique_order = sorted(range(num_cliques), key=lambda i: bin(clique_masks[i]).count('1'), reverse=True)

        def branch_and_bound(uncovered_mask, depth, count):
            if count >= best[0]:
                return
            if uncovered_mask == 0:
                best[0] = count
                return

            # Lower bound
            remaining_edges = bin(uncovered_mask).count('1')
            if remaining_edges == 0:
                best[0] = min(best[0], count)
                return

            # Simple lower bound: each clique can cover at most max_edges_in_clique edges, but more precisely,
            # the largest clique that intersects uncovered_mask
            max_cover = 0
            for i in range(num_cliques):
                covered = bin(clique_masks[i] & uncovered_mask).count('1')
                if covered > max_cover:
                    max_cover = covered
            if max_cover == 0:
                return
            lb = (remaining_edges + max_cover - 1) // max_cover
            if count + lb >= best[0]:
                return

            # Choose an edge to branch on: pick an edge with fewest covering cliques
            # Find the first set bit
            # Pick edge with minimum number of cliques that cover it and are still "available"
            # For simplicity, pick the first uncovered edge
            edge_bit = uncovered_mask & (-uncovered_mask)
            edge_idx = edge_bit.bit_length() - 1

            # Find all cliques that cover this edge
            covering_cliques = []
            for i in range(num_cliques):
                if clique_masks[i] & edge_bit:
                    covering_cliques.append(i)

            if not covering_cliques:
                return

            # Sort by coverage size descending
            covering_cliques.sort(key=lambda i: bin(clique_masks[i] & uncovered_mask).count('1'), reverse=True)

            for i in covering_cliques:
                new_uncovered = uncovered_mask & ~clique_masks[i]
                branch_and_bound(new_uncovered, depth + 1, count + 1)

        branch_and_bound(all_edges_mask, 0, 0)
        return best[0]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        num_cliques = solve_min_clique_cover(n, edges)
        results.append(num_cliques)

    # The task requires returning a list of three lists of cliques (vertex lists), not just counts.
    # Wait, re-reading the task: "construct() -> three lists of cliques (vertex lists), one per instance"
    # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
    # "Minimize the total number of cliques"
    # "The raw value is the total number of cliques."
    # So I need to return the actual cliques, not just the count.

    # I need to modify the solver to also return the actual cliques used.

    # Let me rewrite solve_min_clique_cover to return the list of cliques.

    return results

# Actually, I need to fix this. Let me rewrite properly.

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def get_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        maximal_cliques = []

        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            if P or X:
                u = max(list(P) + list(X), key=lambda x: len(adj[x] & P))
                candidates = P - adj[u]
            else:
                candidates = set()
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bronk(frozenset(), set(range(n)), set())
        return maximal_cliques

    def solve_min_clique_cover(n, edges):
        if not edges:
            return []

        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        num_edges = len(edge_set)
        edge_list = sorted(edge_set)
        edge_index = {e: i for i, e in enumerate(edge_list)}

        max_cliques = get_maximal_cliques(n, list(edge_set))

        clique_edge_sets = []
        for clique in max_cliques:
            verts = list(clique)
            edges_in_clique = set()
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                    if e in edge_set:
                        edges_in_clique.add(e)
            if edges_in_clique:
                clique_edge_sets.append((clique, edges_in_clique))

        if not clique_edge_sets:
            return [[u, v] for u, v in edges]

        num_cliques = len(clique_edge_sets)
        clique_masks = []
        for _, ce in clique_edge_sets:
            mask = 0
            for e in ce:
                mask |= (1 << edge_index[e])
            clique_masks.append(mask)

        all_edges_mask = (1 << num_edges) - 1

        # Greedy for initial upper bound
        uncovered = all_edges_mask
        greedy_cliques = []
        while uncovered:
            best_idx = -1
            best_covered = 0
            for i in range(num_cliques):
                covered = bin(clique_masks[i] & uncovered).count('1')
                if covered > best_covered:
                    best_covered = covered
                    best_idx = i
            if best_idx == -1 or best_covered == 0:
                # Cover remaining edges individually
                for i, e in enumerate(edge_list):
                    if uncovered & (1 << i):
                        greedy_cliques.append([e[0], e[1]])
                break
            uncovered &= ~clique_masks[best_idx]
            greedy_cliques.append(list(clique_edge_sets[best_idx][0]))

        best_solution = greedy_cliques[:]
        best_count = len(best_solution)

        import sys
        sys.setrecursionlimit(10000)

        def branch_and_bound(uncovered_mask, count, current_cliques):
            nonlocal best_solution, best_count
            if count >= best_count:
                return
            if uncovered_mask == 0:
                best_count = count
                best_solution = current_cliques[:]
                return

            remaining_edges = bin(uncovered_mask).count('1')
            if remaining_edges == 0:
                best_count = count
                best_solution = current_cliques[:]
                return

            max_cover = 0
            for i in range(num_cliques):
                covered = bin(clique_masks[i] & uncovered_mask).count('1')
                if covered > max_cover:
                    max_cover = covered
            if max_cover == 0:
                return
            lb = (remaining_edges + max_cover - 1) // max_cover
            if count + lb >= best_count:
                return

            edge_bit = uncovered_mask & (-uncovered_mask)
            covering_cliques = []
            for i in range(num_cliques):
                if clique_masks[i] & edge_bit:
                    covering_cliques.append(i)

            if not covering_cliques:
                return

            covering_cliques.sort(key=lambda i: bin(clique_masks[i] & uncovered_mask).count('1'), reverse=True)

            for i in covering_cliques:
                new_uncovered = uncovered_mask & ~clique_masks[i]
                current_cliques.append(list(clique_edge_sets[i][0]))
                branch_and_bound(new_uncovered, count + 1, current_cliques)
                current_cliques.pop()

        branch_and_bound(all_edges_mask, 0, [])
        return best_solution

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = solve_min_clique_cover(n, edges)
        results.append(cliques)

    return results
