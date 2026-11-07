import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        # Adjacency: C vertices 0..k-1, I vertices k..n-1
        # For each I vertex, which C vertices are adjacent?
        c_adj = {}  # i_vertex -> set of C vertices adjacent
        for v in range(k, n):
            s = set()
            for u in range(k):
                if (min(u, v), max(u, v)) in edge_set:
                    s.add(u)
            c_adj[v] = s

        # C-edges set
        c_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                c_edges.add((u, v))

        cliques = []
        # unassigned C vertices
        unassigned = set(range(k))

        # Sort I vertices by degree (descending) to greedily form large cliques
        i_vertices = sorted(range(k, n), key=lambda v: len(c_adj[v]), reverse=True)

        for v in i_vertices:
            if time.time() > deadline:
                break
            adj = c_adj[v]
            # Find largest subset of unassigned C vertices all in adj
            # Greedy: take all unassigned that are in adj
            candidates = unassigned & adj
            if len(candidates) >= 1:
                # To maximize, take all candidates (they form a clique since all in C)
                # But we want to cover C-edges. Taking all candidates is fine.
                clique = [v] + sorted(candidates)
                if len(clique) >= 2:
                    cliques.append(clique)
                    unassigned -= candidates

        # Cover remaining C-edges with pairs
        remaining_c_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                remaining_c_edges.add((u, v))

        # Remove C-edges already covered by cliques that contain C-vertices
        # A clique covers all pairs within it
        covered = set()
        for cl in cliques:
            for a, b in combinations(cl, 2):
                covered.add((min(a, b), max(a, b)))

        # Remaining C-edges not covered
        uncovered = c_edges - covered
        for u, v in sorted(uncovered):
            cliques.append([u, v])

        # Verify: every edge in exactly one clique
        edge_count = {}
        for cl in cliques:
            if len(cl) < 2:
                raise ValueError("clique too small")
            for a, b in combinations(cl, 2):
                e = (min(a, b), max(a, b))
                edge_count[e] = edge_count.get(e, 0) + 1
                if e not in edge_set:
                    raise ValueError(f"edge {e} not in graph")
        for e in edge_set:
            if edge_count.get(e, 0) != 1:
                raise ValueError(f"edge {e} covered {edge_count.get(e, 0)} times")

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        try:
            cliques = solve_one(n, k, p, seed)
            # Convert to pure Python int lists
            cliques = [[int(x) for x in cl] for cl in cliques]
            results.append(cliques)
        except Exception:
            # Fallback: each edge as a 2-clique (valid but not optimal)
            edges = instance(n, k, p, seed)
            cliques = [[int(u), int(v)] for u, v in edges]
            results.append(cliques)

    return results
