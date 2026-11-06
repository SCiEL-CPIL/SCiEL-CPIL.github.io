import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))
            
        cliques = []
        remaining_edges = set(edge_set)
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Build adjacency for remaining graph
            adj = [set() for _ in range(n)]
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Find a large clique using multiple greedy attempts
            best_clique = []
            
            # Try several starting strategies
            # Strategy 1: Start from highest degree vertex
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Try top few degree vertices as starting points
            sorted_vertices = sorted(range(n), key=lambda x: -deg[x])
            
            for start_idx in range(min(5, len(sorted_vertices))):
                v_start = sorted_vertices[start_idx]
                if not adj[v_start]:
                    continue
                
                # Greedy clique growth with dynamic candidate scoring
                clique = [v_start]
                candidates = set(adj[v_start])
                
                while candidates:
                    # Score each candidate by its degree within the current candidate set
                    # (i.e., how many of its neighbors are also in candidates)
                    best_u = None
                    best_score = -1
                    for u in candidates:
                        score = len(adj[u] & candidates)
                        if score > best_score:
                            best_score = score
                            best_u = u
                    
                    if best_u is None:
                        break
                    
                    clique.append(best_u)
                    candidates = candidates & adj[best_u]
                
                if len(clique) > len(best_clique):
                    best_clique = clique
            
            # Also try a random-ish approach: pick a random edge endpoint
            if len(remaining_edges) > 0:
                # Pick a random edge and try growing from both endpoints
                sample_edge = next(iter(remaining_edges))
                for v_start in [sample_edge[0], sample_edge[1]]:
                    if not adj[v_start]:
                        continue
                    clique = [v_start]
                    candidates = set(adj[v_start])
                    while candidates:
                        best_u = None
                        best_score = -1
                        for u in candidates:
                            score = len(adj[u] & candidates)
                            if score > best_score:
                                best_score = score
                                best_u = u
                        if best_u is None:
                            break
                        clique.append(best_u)
                        candidates = candidates & adj[best_u]
                    if len(clique) > len(best_clique):
                        best_clique = clique
            
            if not best_clique or len(best_clique) < 2:
                # Fallback: just take one edge
                e = next(iter(remaining_edges))
                best_clique = list(e)
            
            # Remove edges in the best clique from remaining
            for u, v in combinations(best_clique, 2):
                e = (u, v) if u < v else (v, u)
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            cliques.append(best_clique)
        
        return cliques

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))
    
    return [c1, c2]
