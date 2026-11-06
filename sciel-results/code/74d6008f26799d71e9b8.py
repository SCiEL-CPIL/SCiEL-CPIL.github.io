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

    def find_large_clique(remaining_edges, adj, n):
        if not remaining_edges:
            return []
        
        # Find a good starting vertex: highest degree in the subgraph defined by remaining_edges
        deg = [0] * n
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
        
        # Start with the vertex with the highest degree
        if max(deg) == 0:
            return []
            
        start_v = max(range(n), key=lambda x: deg[x])
        
        # Greedy maximal clique construction on the remaining graph
        C = [start_v]
        candidates = set()
        for u, v in remaining_edges:
            if u == start_v:
                candidates.add(v)
            elif v == start_v:
                candidates.add(u)
        
        while candidates:
            # Choose candidate that is connected to all in C via REMAINING EDGES
            # and has max connections to other candidates via REMAINING EDGES
            best_cand = None
            best_score = -1
            
            for c in candidates:
                # Check if c is connected to all in C via remaining_edges
                is_connected = True
                for v in C:
                    e = (min(c, v), max(c, v))
                    if e not in remaining_edges:
                        is_connected = False
                        break
                if not is_connected:
                    continue
                
                # Score: number of neighbors in candidates via remaining_edges
                score = 0
                for c2 in candidates:
                    if c != c2:
                        e2 = (min(c, c2), max(c, c2))
                        if e2 in remaining_edges:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand is None:
                break
                
            C.append(best_cand)
            # Update candidates: must be connected to best_cand via remaining_edges
            new_candidates = set()
            for c in candidates:
                if c != best_cand:
                    e = (min(best_cand, c), max(best_cand, c))
                    if e in remaining_edges:
                        new_candidates.add(c)
            candidates = new_candidates
            
        return C

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() - start_time > timeout:
                # Fallback: return whatever we have, plus remaining edges as singletons
                # to ensure validity, though this might be slow to construct
                # Actually, if we time out, we must return a valid solution.
                # The safest valid solution is the initial one (all edges as cliques of size 2).
                # But we have already found some cliques. We can't just return partial.
                # We must complete the partition.
                # If we are close to timeout, we should stop searching for large cliques
                # and just take edges one by one.
                pass
            
            C = find_large_clique(remaining_edges, adj, n)
            
            if not C:
                # Fallback if no clique found (should not happen if edges remain)
                break
            
            # Remove edges in clique C from remaining_edges
            # Since find_large_clique ensures all edges in C are in remaining_edges,
            # we just need to remove them.
            for u, v in combinations(C, 2):
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            cliques.append(C)
            
            # If time is running out, switch to single edge cliques for speed
            if time.time() - start_time > timeout - 5.0:
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break
                
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
