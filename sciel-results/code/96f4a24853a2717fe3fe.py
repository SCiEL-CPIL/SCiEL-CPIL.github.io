import numpy as np
import random
from itertools import combinations

def construct():
    n = 20
    # Use a fixed seed for reproducibility during development, but random for evaluation?
    # The prompt implies we can use random. Let's use a high density random graph.
    # We will use a repair strategy.
    
    # 1. Initialize with a dense random graph
    # p = 0.7 seems reasonable.
    p = 0.7
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # 2. Repair loop
    # Condition: No 3 vertices have 3 common neighbors.
    # To check violation: For every triple (u, v, w), count common neighbors.
    # If count >= 3, violation.
    # To repair: Remove an edge from the set of edges connecting the triple to their common neighbors?
    # Actually, if u, v, w have common neighbors c1, c2, c3, then the edges (u,c1), (v,c1), (w,c1), etc. form K3,3.
    # Removing one edge incident to one of the common neighbors for one of the triple vertices might break it.
    # Specifically, if we pick a common neighbor c, and remove edge (u, c), then c is no longer a common neighbor of u,v,w.
    # So the number of common neighbors decreases by 1.
    
    max_iterations = 5000
    for _ in range(max_iterations):
        violated = False
        # Find a violating triple
        # Optimize: Iterate over triples.
        # To speed up, we can check pairs first?
        # If u, v have >= 3 common neighbors, it doesn't necessarily mean violation (need a 3rd vertex w connected to those).
        # But if u, v, w have >= 3 common neighbors, it's a violation.
        
        found_violation = False
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    # Count common neighbors of u, v, w
                    # Using bitmasks for speed
                    # Let's stick to simple loops for clarity first, or use numpy if available.
                    # Since n=20, n^3 * n = 160,000 operations per check.
                    # 5000 iterations * 160,000 = 800M operations. Might be too slow in pure Python loops.
                    
                    # Let's use a more efficient check.
                    # Precompute neighbor masks?
                    pass
        
        # Alternative: Randomly sample triples to find a violation?
        # Or just check all triples but break early.
        
        # Let's try a simpler repair:
        # Check all triples. If violation found, remove a random edge from the K3,3 subgraph.
        # The K3,3 subgraph involves U={u,v,w} and V={c1,c2,c3}.
        # Edges are all (u', c') for u' in U, c' in V.
        # Removing one such edge breaks this specific K3,3.
        
        violation_found = False
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = []
                    for c in range(n):
                        if c != u and c != v and c != w:
                            if adj[u, c] and adj[v, c] and adj[w, c]:
                                common.append(c)
                    if len(common) >= 3:
                        violation_found = True
                        # Pick 3 common neighbors
                        c1, c2, c3 = common[0], common[1], common[2]
                        # Remove a random edge from the bipartite graph between {u,v,w} and {c1,c2,c3}
                        # There are 9 edges.
                        # Let's remove (u, c1)
                        adj[u, c1] = 0
                        adj[c1, u] = 0
                        break
                if violation_found:
                    break
            if violation_found:
                break
        
        if not violation_found:
            break

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
                
    return edges
