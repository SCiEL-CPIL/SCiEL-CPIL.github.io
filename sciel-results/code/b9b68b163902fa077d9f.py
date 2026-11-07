"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bipartite_check(adj, n):
    """Check if graph is bipartite (triangle-free for this context)."""
    color = [-1] * n
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return False
    return True

def _diameter(adj, n):
    """Compute diameter of unweighted graph."""
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        max_d = max(max_d, max(dist))
    return max_d

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _validate(edge_set, base_edges, n):
    """Check if adding edge_set to base gives valid graph. Return diameter or -1."""
    all_edges = set()
    for u, v in base_edges:
        all_edges.add((min(u, v), max(u, v)))
    for u, v in edge_set:
        if u != v:
            all_edges.add((min(u, v), max(u, v)))
    adj = _build_adj(list(all_edges), n)
    if not _bipartite_check(adj, n):
        return -1
    diam = _diameter(adj, n)
    return diam

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    base_edges = set()
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
    
    # All valid candidate edges (opposite parity, not in base)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (v - u) % 2 == 1 and (u, v) not in base_edges:
                all_candidates.append((u, v))
    
    best_edges = None
    best_count = float('inf')
    
    # Generate diverse candidate sets
    candidate_sets = []
    
    # Strategy 1: Regular step patterns
    for step in range(3, n, 2):  # Only odd steps (opposite parity)
        edges = set()
        for i in range(n):
            j = i + step
            if j < n:
                edges.add((i, j))
        candidate_sets.append(edges)
    
    # Strategy 2: Two-step patterns
    for s1 in range(3, 15, 2):
        for s2 in range(s1 + 2, n, 2):
            edges = set()
            for i in range(n):
                j = i + s1
                if j < n:
                    edges.add((i, j))
                j = i + s2
                if j < n:
                    edges.add((i, j))
            candidate_sets.append(edges)
    
    # Strategy 3: Hub-based
    for h1 in range(2, n - 2):
        for h2 in range(h1 + 2, n - 2):
            if (h2 - h1) % 2 == 0:  # Hubs same parity
                edges = set()
                for v in range(n):
                    if v == h1 or v == h2:
                        continue
                    if (h1 - v) % 2 != 0 and abs(h1 - v) > 2:
                        edges.add((min(h1, v), max(h1, v)))
                    if (h2 - v) % 2 != 0 and abs(h2 - v) > 2:
                        edges.add((min(h2, v), max(h2, v)))
                if edges:
                    candidate_sets.append(edges)
    
    # Strategy 4: Sparse random-like patterns (deterministic)
    for k in [5, 6, 7, 8, 9, 10, 12]:
        edges = set()
        # Select k edges spread across the path
        step = (n - 1) // k
        for i in range(0, n - step, step):
            j = i + step
            if (j - i) % 2 == 1:
                edges.add((i, j))
            # Also add a longer jump
            j2 = i + step * 2
            if j2 < n and (j2 - i) % 2 == 1:
                edges.add((i, j2))
        candidate_sets.append(edges)
    
    # Evaluate candidates
    for edge_set in candidate_sets:
        if time.time() > deadline - 10:
            break
        diam = _validate(edge_set, base_edges, n)
        if diam != -1 and diam <= 4:
            count = len(edge_set)
            if count < best_count:
                best_count = count
                best_edges = set(edge_set)
    
    # If no valid solution, use complete bipartite
    if best_edges is None:
        best_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (v - u) % 2 == 1 and (u, v) not in base_edges:
                    best_edges.add((u, v))
        best_count = len(best_edges)
    
    # Greedy edge removal
    if best_edges and time.time() < deadline - 5:
        current_edges = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for e in sorted(current_edges):
                if time.time() > deadline - 2:
                    break
                trial = current_edges - {e}
                diam = _validate(trial, base_edges, n)
                if diam != -1 and diam <= 4:
                    current_edges = trial
                    best_count = len(trial)
                    improved = True
                    break
        best_edges = current_edges
    
    # Local search: try swapping edges
    if best_edges and time.time() < deadline - 3:
        current_edges = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for e_out in sorted(current_edges):
                if time.time() > deadline - 1:
                    break
                for e_in in all_candidates:
                    if e_in in current_edges:
                        continue
                    trial = (current_edges - {e_out}) | {e_in}
                    diam = _validate(trial, base_edges, n)
                    if diam != -1 and diam <= 4:
                        # Only accept if it doesn't increase count (it won't, since we swap)
                        # But we prefer to keep going for removal opportunities
                        current_edges = trial
                        improved = True
                        break
                if improved:
                    break
        best_edges = current_edges
    
    # Final greedy removal pass
    if best_edges and time.time() < deadline - 1:
        current_edges = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for e in sorted(current_edges):
                if time.time() > deadline - 0.5:
                    break
                trial = current_edges - {e}
                diam = _validate(trial, base_edges, n)
                if diam != -1 and diam <= 4:
                    current_edges = trial
                    improved = True
                    break
        best_edges = current_edges
    
    # Convert to required format
    result = [[u, v] for u, v in sorted(best_edges)]
    result = [[int(u), int(v)] for u, v in result]
    
    return result
# EVOLVE-BLOCK-END
