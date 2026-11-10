import time
import random

def construct():
    n = 15
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Start with K_{7,8}
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    for u in part_a:
        for v in part_b:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    missing_edges = [(u, v) for (u, v) in all_edges if v not in adj[u]]
    
    random.seed(42)
    random.shuffle(missing_edges)
    
    def check_new_violations(u, v, adj, deg):
        """Check if adding edge (u,v) creates any 3-path with equal-degree endpoints."""
        # Case 1: (u,v) is first edge: u-v-y-t, violation if deg(u)==deg(t)
        for y in adj[v]:
            if y == u:
                continue
            for t in adj[y]:
                if t == u or t == v:
                    continue
                if deg[u] == deg[t]:
                    return False
        # Case 2: (u,v) is middle edge: s-u-v-t, violation if deg(s)==deg(t)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        # Case 3: (u,v) is last edge: s-x-u-v, violation if deg(s)==deg(v)
        for x in adj[u]:
            if x == v:
                continue
            for s in adj[x]:
                if s == u or s == v:
                    continue
                if deg[s] == deg[v]:
                    return False
        return True
    
    for (u, v) in missing_edges:
        if time.time() > deadline:
            break
        
        if check_new_violations(u, v, adj, deg):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
    
    result = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
