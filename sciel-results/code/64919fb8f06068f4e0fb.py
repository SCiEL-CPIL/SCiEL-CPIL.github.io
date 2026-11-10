import numpy as np
import time
import random
import sys

def _has_c6_after_add(adj, u, v):
    """Check if adding edge (u,v) to graph with adjacency matrix adj creates a C6.
    A C6 containing (u,v) is a path u-x1-x2-x3-x4-v with all 6 vertices distinct.
    """
    n = adj.shape[0]
    # x1 must be a neighbor of u (not v), x4 must be a neighbor of v (not u)
    # Then we need a path x1-y1-y2-x4 of length 3 with all vertices distinct from {u,v,x1,x4}
    # and x1 != x4, y1 != y2, etc.
    
    # Get neighbors of u (excluding v) and neighbors of v (excluding u)
    nbrs_u = np.where(adj[u] == 1)[0]
    nbrs_v = np.where(adj[v] == 1)[0]
    
    for x1 in nbrs_u:
        if x1 == v:
            continue
        for x4 in nbrs_v:
            if x4 == u or x4 == x1:
                continue
            # Check if there's a path x1-y1-y2-x4 of length 3
            # y1 is neighbor of x1 (not u, not v, not x4), y2 is neighbor of x4 (not u, not v, not x1)
            # and y1-y2 is an edge, y1 != y2, y1 != x4, y2 != x1
            nbrs_x1 = np.where(adj[x1] == 1)[0]
            nbrs_x4 = np.where(adj[x4] == 1)[0]
            
            # y1 in nbrs_x1, y2 in nbrs_x4, adj[y1,y2]==1
            # All of {x1, y1, y2, x4} distinct from {u, v}
            for y1 in nbrs_x1:
                if y1 == u or y1 == v or y1 == x4:
                    continue
                # y2 must be neighbor of x4, neighbor of y1, not u, not v, not x1, not y1
                for y2 in nbrs_x4:
                    if y2 == u or y2 == v or y2 == x1 or y2 == y1:
                        continue
                    if adj[y1, y2] == 1:
                        return True
    return False

def _count_edges(adj):
    return int(np.sum(adj)) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    n = 40
    best_edges = 0
    best_adj = None
    
    # Initial seed: 8 copies of K5
    adj_seed = np.zeros((n, n), dtype=np.int8)
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                adj_seed[base + a, base + b] = 1
                adj_seed[base + b, base + a] = 1
    
    # Try multiple random restarts with greedy edge addition
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            adj = adj_seed.copy()
        else:
            # Random initial graph: start with K5 copies but add some random edges
            adj = adj_seed.copy()
            # Add some random edges
            for _ in range(random.randint(0, 10)):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and adj[u, v] == 0:
                    if not _has_c6_after_add(adj, u, v):
                        adj[u, v] = 1
                        adj[v, u] = 1
        
        # Greedy edge addition
        # Build list of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        
        # Shuffle for diversity
        random.shuffle(all_edges)
        
        # Sort by a heuristic: prefer edges that connect vertices with high degree
        # or use spectral gap heuristic
        # For now, try random order and degree-based order
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add one more edge
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if adj[u, v] == 1:
                    continue
                if not _has_c6_after_add(adj, u, v):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    improved = True
                    # Remove from all_edges (or mark as used)
                    all_edges.remove((u, v))
                    break  # Restart the loop with updated adj
        
        # Also try a second pass with different ordering
        if time.time() < deadline:
            all_edges2 = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
            # Sort by sum of degrees (prefer high-degree connections)
            all_edges2.sort(key=lambda e: adj[e[0]].sum() + adj[e[1]].sum(), reverse=True)
            for u, v in all_edges2:
                if time.time() > deadline:
                    break
                if adj[u, v] == 1:
                    continue
                if not _has_c6_after_add(adj, u, v):
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        edge_count = _count_edges(adj)
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
    
    # Convert best_adj to edge list
    if best_adj is not None:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i, j] == 1:
                    edges.append([i, j])
        return edges
    else:
        # Fallback: return the seed
        edges = []
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([base + a, base + b])
        return edges
