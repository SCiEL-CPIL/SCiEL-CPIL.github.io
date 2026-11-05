"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def solve_instance(adj):
        n = len(adj)
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        if not edges:
            return []
        
        edge_set = set(edges)
        remaining_edges = set(edges)
        
        # Pre-compute all cliques of size >= 2
        # For n=32, 64, this might be large, but we can generate them on the fly or limit size
        # Actually, for dense graphs, large cliques are rare. We should focus on maximal cliques or just greedy expansion.
        # A simpler approach: Greedy set cover.
        # 1. Generate a pool of candidate cliques.
        # To keep it fast, we can just use the edges themselves as candidates initially, 
        # but that's the baseline. We need to find larger cliques.
        
        # Let's find all maximal cliques? No, too expensive for n=64 potentially if dense.
        # Alternative: Greedy construction.
        # Pick an uncovered edge (u,v). Extend it to a maximal clique containing (u,v) using a greedy heuristic.
        # Use that clique. Remove its edges. Repeat.
        
        cliques_found = []
        
        while remaining_edges:
            if time.time() > deadline - 1.0:
                break
            
            # Pick an edge to cover. Maybe pick the one with highest degree sum?
            # For simplicity, just pick the first one, or random.
            # To be deterministic and simple, pick the first from the set.
            # Sets are unordered in Python, so let's convert to list or use a min-heap?
            # For now, just pop an arbitrary one.
            u, v = next(iter(remaining_edges))
            
            # Greedily extend {u, v} to a clique
            # Current clique vertices
            clique_vertices = {u, v}
            
            # Candidate vertices to add: common neighbors of all current vertices
            candidates = adj[u].intersection(adj[v])
            
            # Add vertices one by one that have the most connections to current clique?
            # Or just add any common neighbor?
            # Let's try to add the vertex with the highest degree within the current clique's common neighbors.
            
            while candidates:
                # Pick candidate with highest degree in the graph? Or highest common neighbors?
                # Let's just pick one with max len(adj[w] & candidates) ?
                # Simpler: just pick any. But we want large cliques.
                # Let's sort by degree in the remaining graph?
                # For speed, just pick the first one.
                w = next(iter(candidates))
                clique_vertices.add(w)
                candidates = candidates.intersection(adj[w])
                # Remove edges covered by this new vertex? No, we remove all edges in the final clique at the end.
            
            # Convert clique vertices to list
            clique_list = list(clique_vertices)
            cliques_found.append(clique_list)
            
            # Remove all edges within this clique from remaining_edges
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    edge = (a, b) if a < b else (b, a)
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
            
            # If we ran out of time, we must ensure we don't return invalid result.
            # If remaining_edges is not empty, we have to cover them with single edges to be valid.
            # But we break out of loop. So we handle after loop.
            
        # If we broke early due to time, cover remaining edges as singletons
        if remaining_edges:
            for e in remaining_edges:
                cliques_found.append(list(e))
        
        return cliques_found

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5.0:
            # Not enough time for second instance, return empty or fallback?
            # Must return two lists.
            # If we can't compute, we might fail validity if empty.
            # Let's try to compute as fast as possible.
            pass
            
        adj = random_graph(n, p, seed)
        cliques = solve_instance(adj)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
