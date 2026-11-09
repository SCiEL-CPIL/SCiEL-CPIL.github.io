import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5.0:
             # Emergency fallback: All vertices is a valid vertex cover
             results.append(list(range(n)))
             continue
             
        # 1. Build Graph
        rng = Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        if not edges:
            results.append([])
            continue

        # 2. Greedy Vertex Cover
        # A vertex cover hits every edge, and thus every maximal clique of size >= 2.
        # This guarantees validity.
        current_edges = set(edges)
        cover = set()
        
        while current_edges:
            # Calculate degrees in current graph
            deg = {v: 0 for v in range(n)}
            for u, v in current_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find max degree vertex
            max_v = max(deg, key=deg.get)
            cover.add(max_v)
            
            # Remove edges incident to max_v
            current_edges = { (u,v) for u,v in current_edges if u != max_v and v != max_v }
            
            if time.time() > deadline - 5.0:
                break

        # 3. Local Search: Remove redundant vertices
        # If a vertex in the cover has no edges to vertices outside the cover, it can be removed.
        improved = True
        while improved:
            improved = False
            for v in list(cover):
                # Check if v is essential
                # v is essential if there is an edge (v, u) such that u is NOT in cover
                essential = False
                for u in adj[v]:
                    if u not in cover:
                        essential = True
                        break
                if not essential:
                    cover.remove(v)
                    improved = True
            
            if time.time() > deadline - 5.0:
                break

        results.append(sorted(list(cover)))
        
    return results
