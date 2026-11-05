"""Clique partition of dense random graphs - minimize number of cliques."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(edges, n, start_time, deadline):
        """
        Greedy clique partition.
        Strategy: Repeatedly extract a maximal clique from the remaining edges.
        To improve quality, we try multiple starting vertices and pick the one
        that yields the largest clique.
        """
        if not edges:
            return []

        remaining = set(edges)
        cliques = []

        # Pre-compute full adjacency for fast intersection checks
        full_adj = [set() for _ in range(n)]
        for u, v in edges:
            full_adj[u].add(v)
            full_adj[v].add(u)

        while remaining:
            if time.time() > deadline - 2.0:
                break

            # Build adjacency of the *remaining* graph
            rem_adj = [set() for _ in range(n)]
            rem_vertices = set()
            for u, v in remaining:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                rem_vertices.add(u)
                rem_vertices.add(v)

            if not rem_vertices:
                break

            # Heuristic: Sort vertices by degree in remaining graph
            sorted_verts = sorted(rem_vertices, key=lambda x: len(rem_adj[x]), reverse=True)
            
            best_clique = None
            best_size = 0

            # Try to build a clique starting from top candidates
            # Limit candidates to avoid O(N^2) overhead in large graphs
            candidates_to_try = sorted_verts[:min(30, len(sorted_verts))]
            
            for start in candidates_to_try:
                # Greedy clique expansion
                clique = [start]
                candidates = set(rem_adj[start])
                
                while candidates:
                    # Pick candidate with highest degree within current candidates
                    # This is a standard greedy heuristic for max clique
                    best_cand = max(candidates, key=lambda x: len(rem_adj[x] & candidates))
                    clique.append(best_cand)
                    candidates = candidates & rem_adj[best_cand]
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique

            if best_clique is None or len(best_clique) < 2:
                # Fallback: take any single edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue

            # Add the selected clique
            cliques.append(best_clique)
            
            # Remove all edges covered by this clique from 'remaining'
            clique_set = set(best_clique)
            edges_to_remove = set()
            for u in clique_set:
                for v in clique_set:
                    if u < v:
                        edges_to_remove.add((u, v))
            
            remaining -= edges_to_remove

        # If we broke out early due to deadline, handle remaining edges as 2-cliques
        # This ensures validity (every edge covered) even if suboptimal
        for e in remaining:
            cliques.append(list(e))

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_edges(n, p, seed)
        
        # Allocate time budget for each instance
        # We have ~140s total. 2 instances.
        # Give 60s to each, leaving 20s buffer.
        instance_deadline = _start + 60.0
        
        cliques = greedy_clique_partition(edges, n, _start, instance_deadline)
        
        # Optional: Simple valid refinement
        # We can try to see if any two cliques can be merged.
        # But as analyzed, merging is tricky due to cross-edges.
        # However, if two cliques are vertex-disjoint and the union is a clique,
        # it implies all cross edges exist.
        # If we merge them, we cover cross edges in the new clique.
        # But those cross edges were previously covered by *other* cliques.
        # So we would have double coverage unless we *also* remove those cross edges
        # from the other cliques.
        # This is a complex operation. 
        # For now, we skip refinement to ensure validity and reliability.
        
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
