"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def greedy_clique_from_vertex(v, adj, remaining_edges, n):
        """Greedily build a clique starting from vertex v, only using remaining edges."""
        clique = [v]
        candidates = set(adj[v])
        candidates = {u for u in candidates if (min(v,u), max(v,u)) in remaining_edges}
        
        while candidates:
            best = None
            best_score = -1
            for w in candidates:
                # Check if w connects to all in clique
                valid = True
                for x in clique:
                    e = (min(w,x), max(w,x))
                    if e not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    continue
                # Score: prefer vertices with more common connections
                common = len(adj[w] & set(clique))
                if common > best_score:
                    best_score = common
                    best = w
            if best is not None:
                clique.append(best)
                candidates.remove(best)
            else:
                break
        return clique if len(clique) >= 2 else None

    def branch_and_bound_clique(adj, n, deadline, init_vertices=None):
        """Find a large clique using branch and bound on the complement graph."""
        # Use a simple heuristic: try all starting vertices, extend greedily
        best_clique = init_vertices if init_vertices else []
        best_size = len(best_clique)
        
        for start in range(n):
            if len(best_clique) > 0:
                # Try to extend existing clique
                pass
            # Simple greedy extension from each vertex
            clique = [start]
            candidates = list(adj[start])
            random.shuffle(candidates)
            
            for w in candidates:
                if all((min(w,x), max(w,x)) in adj[w] or w in adj[x] for x in clique):
                    # Check all edges exist
                    valid = True
                    for x in clique:
                        e = (min(w,x), max(w,x))
                        if w not in adj[x]:
                            valid = False
                            break
                    if valid:
                        clique.append(w)
            
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
        
        return best_clique

    def clique_partition(n, p, seed):
        adj = get_graph(n, p, seed)
        remaining_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining_edges.add((u, v))
        
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
            
            # Find a good clique by trying each vertex in remaining edges
            best_clique = None
            best_size = 0
            
            # Get all vertices that appear in remaining edges
            vertices_with_edges = set()
            for (u, v) in remaining_edges:
                vertices_with_edges.add(u)
                vertices_with_edges.add(v)
            
            # Try starting from each vertex, but limit for performance
            vertex_list = list(vertices_with_edges)
            random.shuffle(vertex_list)
            
            # Try a subset of vertices for performance
            max_tries = min(20, len(vertex_list))
            for v in vertex_list[:max_tries]:
                clique = greedy_clique_from_vertex(v, adj, remaining_edges, n)
                if clique and len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or len(best_clique) < 2:
                # Fall back to taking any single edge
                e = next(iter(remaining_edges))
                cliques.append([e[0], e[1]])
                remaining_edges.remove(e)
            else:
                # Remove all edges of this clique
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        u, v = best_clique[i], best_clique[j]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                cliques.append(best_clique)
        
        # Handle remaining edges if we broke early
        if remaining_edges:
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = clique_partition(n, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
