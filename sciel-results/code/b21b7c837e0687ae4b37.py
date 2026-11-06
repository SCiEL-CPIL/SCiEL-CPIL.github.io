import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 120.0
    
    def random_graph(n, p, seed):
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
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        # Helper to check if a set of vertices is a clique
        def is_clique(vertices):
            verts = list(vertices)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if verts[j] not in adj[verts[i]]:
                        return False
            return True

        # Helper to greedily find a large clique starting from a seed vertex
        def greedy_clique(start):
            C = {start}
            candidates = set(adj[start])
            while candidates:
                # Filter candidates to those connected to all in C
                valid_candidates = []
                for v in candidates:
                    connected_to_all = True
                    for u in C:
                        if u not in adj[v]:
                            connected_to_all = False
                            break
                    if connected_to_all:
                        valid_candidates.append(v)
                
                if not valid_candidates:
                    break
                
                # Pick the candidate with the highest degree in the current graph
                # (or just pick one)
                # To be deterministic and fast, pick the one with max len(adj[v])
                best_v = max(valid_candidates, key=lambda x: len(adj[x]))
                C.add(best_v)
                # Update candidates: intersection of candidates and neighbors of best_v
                candidates = candidates.intersection(adj[best_v])
                
            return C

        # Main loop
        while edges:
            if time.time() > deadline - 10:
                # Fallback: just add remaining edges as size-2 cliques
                for u, v in edges:
                    cliques.append([u, v])
                break
            
            # Find a good clique
            best_clique = set()
            # Try starting from a few high-degree vertices
            # Sort vertices by degree descending
            sorted_verts = sorted(range(n), key=lambda x: -len(adj[x]))
            
            # Try top 10 vertices as seeds
            for seed_v in sorted_verts[:10]:
                if not adj[seed_v]:
                    continue
                c = greedy_clique(seed_v)
                if len(c) > len(best_clique):
                    best_clique = c
            
            # If no clique found (shouldn't happen if edges exist, unless isolated vertices which don't have edges)
            # But we only care about edges. If best_clique is empty, pick any edge
            if not best_clique:
                u, v = edges[0]
                best_clique = {u, v}
            
            # Add clique
            cliques.append(list(best_clique))
            
            # Remove edges in this clique from the graph
            # Create a set of edges to remove for fast lookup
            verts = list(best_clique)
            edges_to_remove = set()
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if u > v:
                        u, v = v, u
                    edges_to_remove.add((u, v))
            
            # Remove from edges list and adj
            # This is O(E) which is fine
            new_edges = []
            for e in edges:
                if e not in edges_to_remove:
                    new_edges.append(e)
                else:
                    # Update adj
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
            edges = new_edges
            
        return cliques

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
