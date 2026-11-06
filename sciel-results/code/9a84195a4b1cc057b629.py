"""Clique partition of dense random graphs - greedy maximal clique approach."""
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj
    
    def greedy_clique_partition(edges, adj, n):
        """Greedily partition edges into cliques, minimizing count."""
        uncovered = set(edges)
        cliques = []
        
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            
            # Count remaining edges per vertex
            vertex_count = {}
            for e in uncovered:
                vertex_count[e[0]] = vertex_count.get(e[0], 0) + 1
                vertex_count[e[1]] = vertex_count.get(e[1], 0) + 1
            
            # Try building cliques from high-degree vertices
            sorted_verts = sorted(vertex_count.keys(), key=lambda v: -vertex_count[v])
            
            best_clique = None
            best_size = 0
            
            for start_v in sorted_verts[:30]:
                # Greedily build clique
                clique = [start_v]
                # Candidates: neighbors of start_v that still have uncovered edges
                candidates = [v for v in adj[start_v] if v in vertex_count and v != start_v]
                candidates.sort(key=lambda v: -vertex_count[v])
                
                for v in candidates:
                    if v in clique:
                        continue
                    if all(v in adj[u] for u in clique):
                        clique.append(v)
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or best_size < 2:
                e = next(iter(uncovered))
                best_clique = list(e)
            
            # Remove edges of this clique
            cliques.append(best_clique)
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (min(a, b), max(a, b))
                    uncovered.discard(e)
        
        # Remaining edges as 2-cliques
        for e in uncovered:
            cliques.append(list(e))
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() - start_time > DEADLINE:
            edges, _ = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges, adj = instance(n, p, seed)
        cliques = greedy_clique_partition(edges, adj, n)
        results.append(cliques)
    
    return results
