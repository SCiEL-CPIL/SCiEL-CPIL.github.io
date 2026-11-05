"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 140.0
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets for fast lookup
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert edges to a set of frozensets for quick membership testing
        edge_set = set()
        for u, v in edges:
            edge_set.add(frozenset((u, v)))
        
        # Strategy: Greedily find large cliques
        # 1. Start with all edges
        remaining_edges = set(edge_set)
        cliques = []
        
        # We will iteratively pick a clique, remove its edges, and repeat.
        # To minimize the number of cliques, we want to pick large cliques.
        # Heuristic: Find a vertex with high degree in the remaining graph,
        # then try to extend it to a maximum clique in its neighborhood.
        
        # To speed up, we can pre-compute some structures or just use a simple heuristic.
        # Given n is small (32, 64), we can afford some computation.
        
        # Let's use a simple greedy approach:
        # - While there are edges:
        #   - Find a vertex with the highest degree in the remaining graph.
        #   - Find a maximum clique containing that vertex in the remaining graph.
        #   - Add that clique to the result.
        #   - Remove all edges in that clique from the remaining graph.
        
        # Efficiently finding max clique is NP-hard, but for small n and dense graphs,
        # a heuristic like "expand neighborhood greedily" might work well enough.
        
        # Let's try a different approach:
        # 1. Compute the complement graph.
        # 2. A clique partition in G is an edge cover by cliques.
        # 3. This is equivalent to coloring the complement of the line graph? No.
        # 
        # Let's stick to the greedy clique extraction.
        
        def find_greedy_max_clique(start_vertex, adj_remaining, n):
            # Returns a clique containing start_vertex
            # We use a simple heuristic: sort neighbors by degree, add if compatible
            # This doesn't guarantee max, but finds a reasonably large one.
            # For better performance, we can use a branch and bound for small n, 
            # but let's try the simple one first.
            
            # Actually, let's try to find a maximal clique first, then maybe expand?
            # No, let's just do a greedy expansion.
            
            # Sort neighbors of start_vertex by their degree in the remaining graph
            # descending.
            neighbors = list(adj_remaining[start_vertex])
            # Degree in remaining graph
            deg = {v: len(adj_remaining[v]) for v in range(n)}
            neighbors.sort(key=lambda x: deg[x], reverse=True)
            
            clique = [start_vertex]
            # We need to ensure that every new vertex added is connected to all existing clique members.
            # Since we are adding to a clique, we must check against all current members.
            
            for v in neighbors:
                if all(v in adj_remaining[u] for u in clique):
                    clique.append(v)
                    # Note: We don't update adj_remaining here because we are just finding the clique.
                    # The clique must be a clique in the *original* remaining graph.
                    
            return clique
        
        while remaining_edges:
            # Find vertex with highest degree in remaining graph
            # We need to compute degrees from remaining_edges
            # To make this faster, we can maintain a degree counter.
            # But for simplicity, let's just compute it.
            
            deg = {v: 0 for v in range(n)}
            for e in remaining_edges:
                u, v = e
                deg[u] += 1
                deg[v] += 1
            
            if not any(d > 0 for d in deg.values()):
                break
                
            start_vertex = max(deg, key=lambda x: deg[x])
            
            # Find a greedy clique starting from start_vertex
            # We need to restrict to the subgraph induced by remaining_edges
            # Let's build a local adjacency for the remaining graph
            # This is O(E) per iteration, which might be slow if many iterations.
            # But n is small.
            
            local_adj = [set() for _ in range(n)]
            for e in remaining_edges:
                u, v = e
                local_adj[u].add(v)
                local_adj[v].add(u)
            
            clique = find_greedy_max_clique(start_vertex, local_adj, n)
            
            if len(clique) < 2:
                # Should not happen if there are edges, but just in case
                # Pick an arbitrary edge
                e = next(iter(remaining_edges))
                clique = list(e)
            
            cliques.append(clique)
            
            # Remove edges in this clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = frozenset((clique[i], clique[j]))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        
            if time.time() > deadline - 10:
                # Safety margin
                # If we run out of time, we must return a valid solution.
                # We can't just stop, we must cover all edges.
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append(list(e))
                break
        
        return cliques
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
