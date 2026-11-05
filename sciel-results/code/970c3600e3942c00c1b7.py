"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    """
    Return two lists of cliques (vertex lists), one per instance from
    random_graph(n, p, seed) for (32,0.6,601) and (64,0.55,602).
    Every listed set must be a clique with at least two vertices and
    every edge must lie in exactly one listed clique.
    Minimize the total number of cliques.
    """
    deadline = time.time() + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_clique_partition(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency sets for fast clique checking
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build edge set for tracking
        edge_set = set(edges)
        uncovered = set(edges)

        # Generate candidate cliques using greedy expansion from high-degree vertices
        # Start with all edges as trivial cliques, then try to merge/expand
        # Strategy: find all maximal cliques via Bron-Kerbosch variant or greedy

        # Use a greedy approach to find good cliques:
        # 1. For each vertex, try to expand a clique starting from it
        # 2. Also include all edges as base cliques

        candidates = []

        # Try expanding from each vertex
        for start in range(n):
            # Greedy clique expansion
            clique = [start]
            candidates_to_add = set(adj[start])
            # Sort candidates by degree for better greedy
            candidates_to_add = sorted(candidates_to_add, key=lambda x: len(adj[x]), reverse=True)
            
            for v in candidates_to_add:
                if all(w in adj[v] for w in clique):
                    clique.append(v)
            
            if len(clique) >= 2:
                # Check if this is a valid clique (it should be by construction)
                clique_edges = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if e in edge_set:
                            clique_edges.add(e)
                if clique_edges:
                    candidates.append(clique)

        # Also add all individual edges as candidates (guaranteed valid)
        for u, v in edges:
            candidates.append([u, v])

        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = frozenset(c)
            if key not in seen and len(c) >= 2:
                seen.add(key)
                unique_candidates.append(c)

        # Greedy set cover: pick clique covering most uncovered edges
        selected = []
        while uncovered:
            best_clique = None
            best_count = 0
            for c in unique_candidates:
                # Count how many uncovered edges this clique covers
                count = 0
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        if e in uncovered:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_clique = c
                if count == len(uncovered):
                    break
            
            if best_count == 0 or best_clique is None:
                # Should not happen if we have all edges as candidates
                break
            
            selected.append(best_clique)
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                    uncovered.discard(e)

        # Verify and return
        return selected

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: return trivial edge partition
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        result = solve_clique_partition(n, p, seed)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
