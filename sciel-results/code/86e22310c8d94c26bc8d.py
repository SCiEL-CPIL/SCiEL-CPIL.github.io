import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    # Build base adjacency
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
    
    # Bipartition: even in A, odd in B
    A = set(range(0, n, 2))  # even
    B = set(range(1, n, 2))  # odd
    
    # Candidate edges: between A and B, not already in base
    candidates = []
    for u in A:
        for v in B:
            if frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    # Precompute base adjacency for faster triangle checking
    base_adj = [set() for _ in range(n)]
    for e in base_edges:
        u, v = tuple(e)
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Deadline guard: 90s wall-clock, use 85s for safety
    deadline = time.time() + 85
    
    # Try to find minimum set of edges
    # Start with small counts and increase
    for k in range(1, 10):
        if time.time() > deadline:
            break
        
        # For small k, try all combinations if feasible
        # Number of candidates is ~120 (6*12 - 11 base edges in bipartite)
        # C(120,1) = 120, C(120,2) = 7140, C(120,3) = 280840, C(120,4) = ~8.2M
        # For k<=3, try exhaustive; for k=4, try random; for k>=5, random
        found = None
        
        if k <= 3:
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _check_valid_fast(combo, n, base_edges, base_adj):
                    found = list(combo)
                    break
            if found:
                return [[u, v] for u, v in found]
        else:
            # Random sampling with increasing attempts
            import random
            random.seed(42 + k)
            attempts = 200000 if k <= 5 else 100000
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                combo = random.sample(candidates, k)
                if _check_valid_fast(combo, n, base_edges, base_adj):
                    found = list(combo)
                    break
            if found:
                return [[u, v] for u, v in found]
    
    # Fallback: use the 7-edge solution
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    return [[hub, t] for t in targets]

def _check_valid_fast(added_edges, n, base_edges, base_adj):
    """Fast validity check with early termination."""
    # Build full edge set
    all_edges = set(base_edges)
    for u, v in added_edges:
        all_edges.add(frozenset((u, v)))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for e in all_edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    # Check triangle-free: for each edge (u,v), check no common neighbor
    for e in all_edges:
        u, v = tuple(e)
        common = adj[u] & adj[v]
        if common:
            return False
    
    # Check diameter <= 4 using BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 or d > 4 for d in dist):
            return False
    
    return True

def _check_valid(added_edges, n, base_edges):
    # Build full edge set
    all_edges = set(base_edges)
    for u, v in added_edges:
        all_edges.add(frozenset((u, v)))
    
    # Check triangle-free
    adj = [set() for _ in range(n)]
    for e in all_edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    return False  # triangle found
    
    # Check diameter <= 4
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 or d > 4 for d in dist):
            return False
    
    return True
