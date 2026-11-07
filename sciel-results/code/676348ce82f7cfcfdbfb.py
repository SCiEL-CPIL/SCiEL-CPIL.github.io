"""Triangle-free diameter-four augmentation - improved heuristic search."""
import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            for d in dist:
                if d > max_d:
                    max_d = d
            if max_d > 4:
                return max_d
        return max_d
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                if adj[i] & adj[j]:
                    return False
        return True
    
    def build_adj(added):
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # All candidate edges
    all_candidates = []
    for u in range(n):
        for v in range(u + 3, n):
            all_candidates.append((u, v))
    
    # Compute coverage for each candidate
    far_pairs = [(i, j) for i in range(n) for j in range(i+5, n)]
    
    coverage = []
    for (u, v) in all_candidates:
        count = 0
        for (i, j) in far_pairs:
            d1 = abs(i - u) + 1 + abs(v - j)
            d2 = abs(i - v) + 1 + abs(u - j)
            if min(d1, d2) <= 4:
                count += 1
        coverage.append(count)
    
    # Sort by coverage descending, take top 40
    indexed = list(range(len(all_candidates)))
    indexed.sort(key=lambda i: -coverage[i])
    top_candidates = [all_candidates[i] for i in indexed[:40]]
    
    # Precompute conflicts among top candidates
    def creates_triangle(e1, e2):
        a, b = e1
        c, d = e2
        verts = sorted(set([a, b, c, d]))
        def is_edge(x, y):
            if abs(x - y) == 1:
                return True
            if (min(x,y), max(x,y)) in (e1, e2):
                return True
            return False
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                for k in range(j+1, len(verts)):
                    v1, v2, v3 = verts[i], verts[j], verts[k]
                    if is_edge(v1, v2) and is_edge(v2, v3) and is_edge(v1, v3):
                        return True
        return False
    
    m = len(top_candidates)
    conflict = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if creates_triangle(top_candidates[i], top_candidates[j]):
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Map from edge to index in top_candidates
    edge_to_idx = {e: i for i, e in enumerate(top_candidates)}
    
    deadline = time.time() + 85
    
    best_added = None
    best_len = m + 1
    
    def greedy(start_order):
        adj = [set(base_adj[i]) for i in range(n)]
        added = []
        added_set = set()
        
        for idx in start_order:
            if time.time() > deadline:
                break
            u, v = top_candidates[idx]
            if (u, v) in added_set:
                continue
            # Check conflict with added
            skip = False
            for sel_idx in added_set:
                if idx in conflict[sel_idx]:
                    skip = True
                    break
            if skip:
                continue
            if adj[u] & adj[v]:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            added.append(idx)
            added_set.add(idx)
        
        return added, adj
    
    def local_search(added_indices, adj):
        nonlocal best_added, best_len
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try deleting an edge
            for i in range(len(added_indices) - 1, -1, -1):
                trial = added_indices[:i] + added_indices[i+1:]
                trial_adj = build_adj([top_candidates[j] for j in trial])
                if is_triangle_free(trial_adj) and bfs_diameter(trial_adj) <= 4:
                    added_indices = trial
                    adj = trial_adj
                    improved = True
                    break
            
            # Try swapping: remove one, add one
            if not improved:
                for i in range(len(added_indices)):
                    for j in range(m):
                        if j in added_indices:
                            continue
                        # Check if j conflicts with any in added_indices except i
                        skip = False
                        for k in added_indices:
                            if k == i:
                                continue
                            if j in conflict[k]:
                                skip = True
                                break
                        if skip:
                            continue
                        
                        u, v = top_candidates[j]
                        # Check triangle-free when adding j to adj without edge i
                        # Temporarily remove edge i
                        ui, vi = top_candidates[added_indices[i]]
                        adj[ui].discard(vi)
                        adj[vi].discard(ui)
                        
                        if adj[u] & adj[v]:
                            adj[ui].add(vi)
                            adj[vi].add(ui)
                            continue
                        
                        # Add j
                        adj[u].add(v)
                        adj[v].add(u)
                        new_added = [k for k in added_indices if k != added_indices[i]] + [j]
                        
                        if is_triangle_free(adj) and bfs_diameter(adj) <= 4:
                            added_indices = new_added
                            improved = True
                            break
                        else:
                            adj[u].discard(v)
                            adj[v].discard(u)
                        
                        adj[ui].add(vi)
                        adj[vi].add(ui)
                        
                        if improved:
                            break
                    if improved:
                        break
        
        if len(added_indices) < best_len:
            best_len = len(added_indices)
            best_added = list(added_indices)
        
        return added_indices, adj
    
    # Run multiple restarts with different orderings
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        order = list(range(m))
        random.shuffle(order)
        # Bias toward high coverage
        order.sort(key=lambda i: (random.random(), -coverage[i]))
        
        added, adj = greedy(order)
        added, adj = local_search(added, adj)
    
    if best_added is None:
        return []
    
    added_edges = [top_candidates[idx] for idx in best_added]
    final_adj = build_adj(added_edges)
    
    assert is_triangle_free(final_adj), "Triangle found"
    assert bfs_diameter(final_adj) <= 4, "Diameter too large"
    
    return [[int(u), int(v)] for u, v in added_edges]
