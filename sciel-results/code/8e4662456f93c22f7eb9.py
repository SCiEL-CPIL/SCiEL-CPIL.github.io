"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def _check_valid(adj, n):
    """Check if no 3-path has endpoints of equal degree."""
    deg = [adj[i].__len__() for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _add_edge(adj, n, u, v):
    adj[u].add(v)
    adj[v].add(u)

def _remove_edge(adj, n, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def construct():
    n = 15
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Start with K_{7,8}
    adj = [set() for _ in range(n)]
    A = list(range(7))
    B = list(range(7, 15))
    for u in A:
        for v in B:
            _add_edge(adj, n, u, v)
    
    # Try to add intra-partition edges greedily
    # Generate candidate edges in order of "safety"
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                continue
            candidates.append((i, j))
    
    # Sort by: prefer edges between vertices that will create distinct degree classes
    # Actually, just try all and check
    for u, v in candidates:
        if time.time() > deadline:
            break
        _add_edge(adj, n, u, v)
        if _check_valid(adj, n):
            pass  # keep it
        else:
            _remove_edge(adj, n, u, v)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
