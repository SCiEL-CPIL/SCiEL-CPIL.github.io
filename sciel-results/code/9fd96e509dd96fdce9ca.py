import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(edges, n):
        if not edges:
            return []
        
        # Build adjacency from edges
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(edges)
        cliques = []
        
        while remaining:
            if time.time() > DEADLINE:
                # Fallback: add remaining edges as individual cliques
                for e in remaining:
                    cliques.append(list(e))
                break
            
            # Find a good clique in the remaining edge set
            best_clique = None
            best_size = 0
            
            # Try building cliques from each vertex
            for start in range(n):
                # Current clique: just {start}
                clique = {start}
                # Candidates: neighbors of start in remaining graph
                candidates = list(adj[start])
                
                # Greedily add vertices to clique
                i = 0
                while i < len(candidates):
                    v = candidates[i]
                    # Check if v is connected to all current clique members in remaining edges
                    can_add = True
                    for u in clique:
                        if (min(u, v), max(u, v)) not in remaining:
                            can_add = False
                            break
                    if can_add:
                        clique.add(v)
                    else:
                        candidates.pop(i)
                    i += 1
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or len(best_clique) < 2:
                # No clique of size >= 2 found; take a single edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
            else:
                # Remove all edges within best_clique from remaining
                clique_list = list(best_clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        u, v = clique_list[i], clique_list[j]
                        e = (min(u, v), max(u, v))
                        if e in remaining:
                            remaining.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                cliques.append(clique_list)
        
        return cliques

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        return greedy_clique_partition(edges, n)

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
