import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
        return edges, adj

    def greedy_clique(start_v, candidates_init, adj, n):
        C = [start_v]
        candidates = set(candidates_init)
        
        while candidates:
            # Choose candidate that is connected to all in C and has max connections to other candidates
            best_cand = None
            best_score = -1
            for c in candidates:
                score = 0
                for c2 in candidates:
                    if c != c2 and adj[c][c2]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand is None:
                break
                
            C.append(best_cand)
            new_candidates = set()
            for c in candidates:
                if c != best_cand and adj[best_cand][c]:
                    new_candidates.add(c)
            candidates = new_candidates
            
        return C

    def find_large_clique(edges, adj, n):
        if not edges:
            return []
        
        # Find vertices with high degree in the remaining subgraph
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        # Get top 10 vertices by degree
        top_vertices = sorted(range(n), key=lambda x: deg[x], reverse=True)[:10]
        
        best_clique = []
        
        for start_v in top_vertices:
            # Candidates for start_v
            candidates_init = set()
            for u, v in edges:
                if u == start_v:
                    candidates_init.add(v)
                elif v == start_v:
                    candidates_init.add(u)
            
            C = greedy_clique(start_v, candidates_init, adj, n)
            
            if len(C) > len(best_clique):
                best_clique = C
                
        return best_clique

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() - start_time > timeout - 5.0:
                # Fallback: add remaining edges as singletons
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break
            
            C = find_large_clique(remaining_edges, adj, n)
            
            if not C:
                # Should not happen, but handle gracefully
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break
            
            # Remove edges in clique C from remaining_edges
            for u, v in combinations(C, 2):
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            cliques.append(C)
            
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
