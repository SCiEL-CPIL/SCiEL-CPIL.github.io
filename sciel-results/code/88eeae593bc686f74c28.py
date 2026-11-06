import random
import time
from collections import defaultdict

def is_triangle_free_edge(adj, n, u, v):
    if v in adj[u]:
        return False
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def is_3_colorable(adj, n):
    color = [-1] * n
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        used = set()
        for w in adj[v]:
            if color[w] != -1:
                used.add(color[w])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def make_grotzsch_edges():
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
    for i in range(5):
        edges.append([10, 5 + i])
    return edges

def construct():
    n = 16
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            
    grotzsch_edges = make_grotzsch_edges()
    grotzsch_set = set(tuple(sorted(e)) for e in grotzsch_edges)
    
    initial_adj = [set() for _ in range(n)]
    for e in grotzsch_edges:
        u, v = e
        initial_adj[u].add(v)
        initial_adj[v].add(u)
        
    remaining_edges = [e for e in all_edges if tuple(sorted(e)) not in grotzsch_set]
    
    # Heuristic ordering: try edges that are "central" first?
    # Or just random?
    # Let's try sorting by the sum of degrees of endpoints in the initial graph.
    # High degree connections might be more likely to be part of a dense solution?
    # Or low degree to avoid blocking?
    # Let's try sorting by potential conflicts.
    
    def conflict_count(e):
        u, v = e
        # Count how many other remaining edges form a triangle with e
        count = 0
        for w in range(n):
            if w != u and w != v:
                if (u, w) in grotzsch_set or (w, v) in grotzsch_set:
                    continue # Already handled or not in remaining?
                # Check if (u,w) and (w,v) are in remaining or grotzsch
                # If we add (u,v), we block (u,w) and (w,v) if they are selected?
                # No, we block the triangle (u,v,w).
                # So if (u,w) is selected, we cannot select (w,v).
                # This is complex.
        return 0
        
    # Simple sort
    random.seed(42)
    random.shuffle(remaining_edges)
    
    best_count = len(grotzsch_edges)
    best_adj = [set(s) for s in initial_adj]
    
    start_time = time.time()
    time_limit = 100 # seconds
    
    def solve(idx, adj, count):
        nonlocal best_count, best_adj, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if count + len(remaining_edges) - idx <= best_count:
            return
            
        if idx == len(remaining_edges):
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
            return
        
        u, v = remaining_edges[idx]
        
        # Try adding
        if is_triangle_free_edge(adj, n, u, v):
            adj[u].add(v)
            adj[v].add(u)
            solve(idx + 1, adj, count + 1)
            adj[u].remove(v)
            adj[v].remove(u)
            
        # Try skipping
        solve(idx + 1, adj, count)

    solve(0, initial_adj, len(grotzsch_edges))
    
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges
