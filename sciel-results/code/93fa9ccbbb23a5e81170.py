import random
import numpy as np

def construct():
    # Try to construct a C4-free graph on n=39 with alpha < 11
    # using a greedy random approach
    
    target_n = 39
    
    def is_c4_free(adj, u, v):
        # Check if adding edge (u,v) creates a C4
        # C4 exists if there is a vertex w such that adj[u][w] and adj[v][w]
        # AND there is another vertex x such that adj[u][x] and adj[v][x]?
        # No, adding (u,v) creates a C4 if there is a path of length 3 between u and v?
        # i.e. u-w-x-v.
        # This means w is neighbor of u, x is neighbor of v, and w-x is an edge?
        # Actually, a C4 is u-v-w-x-u.
        # If we add (u,v), we create a C4 if there is a path u-a-b-v where a,b are distinct and not u,v.
        # This requires a neighbor a of u and neighbor b of v such that a-b is an edge.
        # AND a != b.
        
        # Simpler: Count common neighbors of u and v.
        # If they already have >= 1 common neighbor, adding (u,v) creates a triangle? No.
        # If they have a common neighbor w, then u-w-v is a path of length 2.
        # Adding (u,v) creates a triangle u-v-w-u.
        # We don't care about triangles.
        # We care about C4.
        # A C4 is a cycle of length 4.
        # Adding (u,v) creates a C4 if there is a path of length 3 from u to v.
        # Path: u-a-b-v.
        # a is neighbor of u.
        # b is neighbor of v.
        # a-b is an edge.
        # a != b.
        
        neighbors_u = [i for i in range(target_n) if adj[u][i]]
        neighbors_v = [i for i in range(target_n) if adj[v][i]]
        
        for a in neighbors_u:
            for b in neighbors_v:
                if a == b:
                    continue
                if adj[a][b]:
                    return False # C4 found
        
        return True

    def has_independent_set(adj, k):
        # Backtracking
        n = target_n
        def backtrack(start, chosen):
            if len(chosen) == k:
                return True
            if len(chosen) + (n - start) < k:
                return False
            for i in range(start, n):
                if all(not adj[i][j] for j in chosen):
                    if backtrack(i+1, chosen+[i]):
                        return True
            return False
        return backtrack(0, [])

    # Try multiple times
    for attempt in range(100):
        adj = np.zeros((target_n, target_n), dtype=bool)
        edges = []
        
        # Random edge order
        all_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_c4_free(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
                edges.append([u, v])
                
        # Check independence
        if not has_independent_set(adj, 11):
            return {'n': target_n, 'edges': edges}
            
    # Fallback to n=31 Polarity Graph
    # (Copy current code)
    
