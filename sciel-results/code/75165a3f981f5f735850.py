"""Spectral construction + local swaps for C6-free graphs on 40 vertices."""
import numpy as np
import random

def _has_c6(adj, n):
    """Check if graph contains a C6. Returns True if C6 found."""
    adj_list = [list(np.where(adj[i])[0]) for i in range(n)]
    
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                continue
            # Find all length-3 paths from u to v
            paths = []
            for a in adj_list[u]:
                if a == v:
                    continue
                for c in adj_list[v]:
                    if c == u or c == a:
                        continue
                    for b in adj_list[a]:
                        if b in (u, v, a, c):
                            continue
                        if adj[b, c]:
                            paths.append((a, b, c))
            if len(paths) >= 2:
                for i in range(len(paths)):
                    for j in range(i+1, len(paths)):
                        s1 = {paths[i][0], paths[i][1], paths[i][2]}
                        s2 = {paths[j][0], paths[j][1], paths[j][2]}
                        if not (s1 & s2):
                            return True
    return False

def construct():
    n = 40
    
    # Start with 4-regular circulant: i±1, i±2 mod 40
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for d in [1, 2]:
            j = (i + d) % n
            adj[i, j] = True
            adj[j, i] = True
    
    # Verify C6-free, remove edges if needed
    while _has_c6(adj, n):
        # Find an edge in a C6 and remove it
        adj_list = [list(np.where(adj[i])[0]) for i in range(n)]
        removed = False
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    continue
                # Check if removing (u,v) helps
                adj[u, v] = False
                adj[v, u] = False
                if not _has_c6(adj, n):
                    removed = True
                    break
                else:
                    adj[u, v] = True
                    adj[v, u] = True
            if removed:
                break
        if not removed:
            break
    
    # Greedy add edges
    random.seed(42)
    candidates = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u, v]]
    random.shuffle(candidates)
    
    for u, v in candidates:
        adj[u, v] = True
        adj[v, u] = True
        if _has_c6(adj, n):
            adj[u, v] = False
            adj[v, u] = False
    
    # Local swaps
    for _ in range(100):  # limited iterations
        improved = False
        random.shuffle(candidates)
        for u, v in candidates:
            if adj[u, v]:
                continue
            # Try adding (u,v) and removing some other edge
            for w in range(n):
                if w == u or w == v:
                    continue
                for x in range(w+1, n):
                    if x == u or x == v:
                        continue
                    if not adj[w, x]:
                        continue
                    # Try swap: add (u,v), remove (w,x)
                    adj[u, v] = True
                    adj[v, u] = True
                    adj[w, x] = False
                    adj[x, w] = False
                    if not _has_c6(adj, n):
                        improved = True
                        break
                    else:
                        adj[u, v] = False
                        adj[v, u] = False
                        adj[w, x] = True
                        adj[x, w] = True
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    return edges
