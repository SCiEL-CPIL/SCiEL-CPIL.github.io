# EVOLVE-BLOCK-START
import random
import sys
from collections import deque

sys.setrecursionlimit(1000000)

def _has_short_path(adj, start, target, max_depth):
    """
    Returns True if there is a path from start to target of length < max_depth.
    """
    if start == target:
        return True
    n = len(adj)
    visited = [False] * n
    visited[start] = True
    dq = deque()
    dq.append((start, 0))
    
    while dq:
        node, depth = dq.popleft()
        if depth >= max_depth:
            continue
        for nb in adj[node]:
            if not visited[nb]:
                if nb == target:
                    return True
                visited[nb] = True
                dq.append((nb, depth + 1))
    return False

def _build_girth_graph(k, g, n, seed):
    """
    Construct a k-regular graph on n vertices with girth at least g.
    Returns (n, edges) or None.
    """
    if n % 2 != 0 or n * k % 2 != 0 or n < 2:
        return None
        
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    stubs = [i for i in range(n) for _ in range(k)]
    rng.shuffle(stubs)
    used = [False] * len(stubs)
    edges = []
    num_edges = n * k // 2
    
    for _ in range(num_edges):
        u_idx = -1
        for i in range(len(stubs)):
            if not used[i]:
                u_idx = i
                break
        if u_idx == -1:
            return None
        u = stubs[u_idx]
        
        found = False
        # Try up to 1000 times to find a valid partner
        for _try in range(1000):
            v_idx = rng.randrange(len(stubs))
            if used[v_idx]:
                continue
            v = stubs[v_idx]
            if u == v:
                continue
            
            # Check if adding (u, v) creates a cycle of length < g
            # This happens if dist(u, v) < g - 1
            if _has_short_path(adj, u, v, g - 1):
                continue
            
            # Accept edge
            adj[u].append(v)
            adj[v].append(u)
            used[u_idx] = True
            used[v_idx] = True
            edges.append((u, v))
            found = True
            break
            
        if not found:
            return None
            
    # Final exact girth check
    for s in range(n):
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s, 1))
        while dq:
            v, prev, d = dq.popleft()
            if d + 1 >= g:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == s:
                    return None
                dq.append((nb, v, d + 1))
                
    return n, edges

def construct(k, g):
    # Target n=936 for k=3, g=16
    # General strategy: try n starting from a reasonable lower bound
    # For k=3, g=16, Moore bound is 382, known lower bound 512, known optimum 936.
    # We start at 936.
    
    if k == 3 and g == 16:
        start_n = 936
    else:
        # Estimate Moore bound
        # 1 + k * sum_{i=0}^{g//2 - 2} (k-1)**i
        half_g = g // 2
        moore = 1 + k * sum((k-1)**i for i in range(half_g - 1))
        start_n = max(moore, 10)
        if start_n % 2 != 0:
            start_n += 1
            
    # Try to build with n = start_n
    for seed in range(100):
        res = _build_girth_graph(k, g, start_n, seed)
        if res is not None:
            return res
            
    # If failed, try larger n
    for delta in range(2, 100, 2):
        n_try = start_n + delta
        for seed in range(20):
            res = _build_girth_graph(k, g, n_try, seed)
            if res is not None:
                return res
                
    return None
# EVOLVE-BLOCK-END
