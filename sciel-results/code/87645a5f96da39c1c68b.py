import time
import random

def construct():
    deadline = time.time() + 50  # 50s safety margin
    
    n = 15
    # Start with K_{7,8}: vertices 0-6 in A, 7-14 in B
    adj = [[] for _ in range(n)]
    deg = [0] * n
    edges = []
    
    # Build K_{7,8}
    for u in range(7):
        for v in range(7, 15):
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            edges.append([u, v])
    
    def can_add(u, v):
        # Check if adding edge (u,v) violates the constraint
        # For any s in adj[u] (s != v) and t in adj[v] (t != u): deg(s) != deg(t)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if deg[s] == deg[t]:
                    return False
        return True
    
    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])
    
    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1
        edges.remove([u, v])
    
    # Try to add edges within A (0-6) and within B (7-14)
    # Generate all candidate edges within parts
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
    
    # Greedy: try to add as many as possible
    random.shuffle(candidates)
    added = 0
    for u, v in candidates:
        if time.time() > deadline:
            break
        if can_add(u, v):
            add_edge(u, v)
            added += 1
    
    # Try a second pass with different ordering
    remaining = [(u, v) for u, v in candidates if [u, v] not in edges and [v, u] not in edges]
    random.shuffle(remaining)
    for u, v in remaining:
        if time.time() > deadline:
            break
        if can_add(u, v):
            add_edge(u, v)
            added += 1
    
    # Try local search: remove an edge and see if we can add more
    # But for simplicity, just return what we have
    # Ensure return type is correct
    result = [[int(u), int(v)] for u, v in edges]
    return result
