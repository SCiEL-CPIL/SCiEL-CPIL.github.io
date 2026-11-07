"""Rewritten program for triangle-free diameter-four augmentation."""
import time
from collections import deque

# EVOLVE-BLOCK-START

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter(adj):
    """Compute diameter via BFS from each node."""
    n = len(adj)
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    q.append(v)
        # Check connectivity
        if -1 in dist:
            return n  # disconnected
    return max_dist

def _check_triangle_free(adj, u, v):
    """Check if adding edge (u,v) would create a triangle."""
    if adj[u] & adj[v]:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    n = P['n']
    base_edges = set()
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_edges.add((min(u,v), max(u,v)))
        base_edges.add((max(u,v), min(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    # Generate all candidate edges (non-base, undirected)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Try with 0 added edges
    d0 = _bfs_diameter(adj)
    if d0 <= 4:
        return []
    
    # Greedy: iteratively add the edge that most reduces diameter
    cur_adj = [set(s) for s in adj]
    added = []
    current_diam = d0
    
    for step in range(12):
        if time.time() > deadline:
            break
        if current_diam <= 4:
            break
        
        best_edge = None
        best_diam = current_diam
        
        for (u, v) in candidates:
            if (u, v) in base_edges:
                continue
            # Skip if already added
            if v in cur_adj[u]:
                continue
            # Check triangle-free
            if cur_adj[u] & cur_adj[v]:
                continue
            
            # Try adding
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            d = _bfs_diameter(cur_adj)
            cur_adj[u].discard(v)
            cur_adj[v].discard(u)
            
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        cur_adj[u].add(v)
        cur_adj[v].add(u)
        added.append((u, v))
        current_diam = best_diam
    
    # Final check
    if _bfs_diameter(cur_adj) <= 4:
        return [[int(u), int(v)] for u, v in added]
    
    # Fallback: try a more aggressive approach with specific known-good edges
    # For a path of 24 nodes, we need to reduce diameter from 23 to 4
    # This requires adding edges that create shortcuts
    # A good strategy is to add edges connecting nodes that are far apart
    # but ensure no triangles are created
    
    # Reset and try a structured approach
    cur_adj2 = [set(s) for s in adj]
    added2 = []
    
    # Add edges in a pattern that reduces diameter while maintaining triangle-free
    # For a path, adding edges (i, j) where j - i > 1 doesn't create triangles
    # if we're careful about which edges we add
    
    # Strategy: add edges to create a "ladder" or "grid" like structure
    # Connect every k-th node to create shortcuts
    
    # Try adding edges that connect nodes at distance 5, 10, 15, 20
    # But avoid creating triangles
    
    for target_dist in [6, 5, 4]:
        for i in range(n):
            j = i + target_dist
            if j >= n:
                continue
            if (i, j) in base_edges:
                continue
            if j in cur_adj2[i]:
                continue
            if cur_adj2[i] & cur_adj2[j]:
                continue
            cur_adj2[i].add(j)
            cur_adj2[j].add(i)
            added2.append((i, j))
        
        if _bfs_diameter(cur_adj2) <= 4:
            break
    
    if _bfs_diameter(cur_adj2) <= 4:
        return [[int(u), int(v)] for u, v in added2]
    
    # Final fallback: return empty list (will be invalid but avoids timeout)
    return []

# EVOLVE-BLOCK-END
