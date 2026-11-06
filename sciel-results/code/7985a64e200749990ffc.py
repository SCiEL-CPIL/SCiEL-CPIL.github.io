import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 40
    
    def check_c6_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6 using bitset operations.
        A new C6 must use the new edge (u,v).
        Path: u - a - b - c - d - v
        We need: a in N(u), d in N(v), and a path of length 3 from a to d
        that avoids u, v and has distinct internal vertices.
        """
        # Use integer bitmasks for fast set operations
        Nu_mask = adj[u]
        Nv_mask = adj[v]
        
        # Get neighbor lists as lists for iteration
        Nu = [i for i in range(n) if Nu_mask[i] and i != v]
        Nv = [i for i in range(n) if Nv_mask[i] and i != u]
        
        for a in Nu:
            Na = [b for b in range(n) if adj[a, b] and b not in (u, v, a)]
            for d in Nv:
                if d == a:
                    continue
                Nd = adj[d]
                # Need b in Na and c in N(b) ∩ N(d), with c not in {u, v, a, b, d}
                for b in Na:
                    if b == d:
                        continue
                    # Find common neighbors of b and d, excluding u, v, a, b, d
                    common = adj[b] & Nd
                    # Remove u, v, a, b, d from consideration
                    common[u] = False
                    common[v] = False
                    common[a] = False
                    common[b] = False
                    common[d] = False
                    if common.any():
                        return True
        return False
    
    # Start with a structured seed: 8 copies of K5
    edges = []
    adj = np.zeros((n, n), dtype=bool)
    
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u, v = 5*k+a, 5*k+b
                adj[u, v] = adj[v, u] = True
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try to add all possible cross edges
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                continue
            if time.time() > deadline:
                return best_edges
            if not check_c6_with_edge(adj, i, j):
                adj[i, j] = adj[j, i] = True
                edges.append([i, j])
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
    
    # Second pass: try random orderings to potentially find more edges
    remaining = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                remaining.append((i, j))
    
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        random.shuffle(remaining)
        for i, j in remaining:
            if time.time() > deadline:
                break
            if not adj[i, j]:
                if not check_c6_with_edge(adj, i, j):
                    adj[i, j] = adj[j, i] = True
                    edges.append([i, j])
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = list(edges)
        if len(edges) == best_count:
            break
    
    return best_edges
