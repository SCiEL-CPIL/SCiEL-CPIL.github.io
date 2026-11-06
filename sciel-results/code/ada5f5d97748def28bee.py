"""Construct a graph on 31 vertices with 91 edges minimizing 4-cycles."""
import time
import numpy as np
from itertools import combinations

def _count_c4(adj):
    """Count 4-cycles in an undirected graph given by adjacency matrix."""
    n = adj.shape[0]
    common = (adj @ adj).astype(int)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = common[i, j]
            if c >= 2:
                total += c * (c - 1) // 2
    return total // 2

def _delta_c4_add_edge(adj, u, v):
    """Compute number of new C4s created by adding edge (u,v)."""
    n = adj.shape[0]
    # Common neighbors of u and v (excluding each other)
    cn = 0
    for k in range(n):
        if k != u and k != v and adj[u, k] and adj[v, k]:
            cn += 1
    return cn * (cn - 1) // 2

def _delta_c4_remove_edge(adj, u, v):
    """Compute number of C4s destroyed by removing edge (u,v)."""
    n = adj.shape[0]
    # C4s using edge (u,v): for each pair of common neighbors (w1,w2) of u and v,
    # if w1-w2 is NOT an edge, then u-w1-v-w2-u is a C4 that uses edge (u,v).
    # Wait, actually any C4 containing edge (u,v) has the form u-w1-v-w2-u
    # where w1,w2 are common neighbors of u and v. The number of such C4s is
    # C(cn(u,v), 2) but we need to be careful: we're counting C4s that have
    # edge (u,v) as one of their 4 edges.
    # Each such C4 is uniquely determined by a pair {w1,w2} of common neighbors.
    # But we must ensure w1 != w2 and all 4 vertices distinct.
    cn = 0
    for k in range(n):
        if k != u and k != v and adj[u, k] and adj[v, k]:
            cn += 1
    return cn * (cn - 1) // 2

def _polarity_graph_q5():
    """Construct the polarity graph of PG(2,5). C4-free, 31 vertices, 90 edges."""
    points = []
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    for b in range(5):
        points.append((0, 1, b))
    points.append((0, 0, 1))
    
    n = len(points)
    assert n == 31
    
    def bilinear(p, q):
        x, y, z = p
        xp, yp, zp = q
        return (x * zp + xp * z + 2 * y * yp) % 5
    
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if bilinear(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return adj, edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Step 1: Get polarity graph (C4-free, 90 edges)
    adj, edges = _polarity_graph_q5()
    
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Step 2: Try all single-edge additions, pick the best
    best_add_edge = None
    best_add_c4 = float('inf')
    for i in range(31):
        for j in range(i + 1, 31):
            if (i, j) in edge_set:
                continue
            c4s = _delta_c4_add_edge(adj, i, j)
            if c4s < best_add_c4:
                best_add_c4 = c4s
                best_add_edge = (i, j)
    
    # Add the best edge
    u, v = best_add_edge
    adj[u, v] = 1
    adj[v, u] = 1
    edges.append([u, v])
    edge_set.add((u, v))
    edge_set.add((v, u))
    
    current_c4 = _count_c4(adj)
    
    # Step 3: Local search via edge swaps to reduce C4 count
    # Try removing one edge and adding another (keeping 91 edges)
    # Delta = C4s_lost_from_removal - C4s_gained_from_addition
    # We want to maximize this delta (i.e., minimize net C4 change)
    
    max_iterations = 50000
    iteration = 0
    improved = True
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        best_delta = 0  # Only accept strictly improving moves
        best_swap = None  # (remove_u, remove_v, add_u, add_v)
        
        current_edges = list(edge_set)
        # Build list of directed edges for iteration
        undirected_edges = [(a, b) for a, b in edge_set if a < b]
        
        for rem_u, rem_v in undirected_edges:
            if time.time() > deadline:
                break
            # Compute C4s destroyed by removing (rem_u, rem_v)
            c4_destroyed = _delta_c4_remove_edge(adj, rem_u, rem_v)
            
            if c4_destroyed <= 0:
                continue
            
            # Try adding each non-edge
            for add_u in range(31):
                if add_u == rem_u or add_u == rem_v:
                    continue
                for add_v in range(add_u + 1, 31):
                    if add_v == rem_u or add_v == rem_v:
                        continue
                    if (add_u, add_v) in edge_set:
                        continue
                    if (rem_u, rem_v) == (add_u, add_v) or (rem_v, rem_u) == (add_u, add_v):
                        continue
                    
                    c4_added = _delta_c4_add_edge(adj, add_u, add_v)
                    delta = c4_destroyed - c4_added
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (rem_u, rem_v, add_u, add_v)
        
        if best_swap is not None:
            rem_u, rem_v, add_u, add_v = best_swap
            # Perform the swap
            adj[rem_u, rem_v] = 0
            adj[rem_v, rem_u] = 0
            edge_set.discard((rem_u, rem_v))
            edge_set.discard((rem_v, rem_u))
            
            adj[add_u, add_v] = 1
            adj[add_v, add_u] = 1
            edge_set.add((add_u, add_v))
            edge_set.add((add_v, add_u))
            
            # Update edges list
            edges = [[a, b] for a, b in edge_set if a < b]
            current_c4 += (-best_delta)  # best_delta is improvement
            improved = True
            iteration += 1
    
    # Rebuild edges list from edge_set
    edges = [[a, b] for a, b in edge_set if a < b]
    edges.sort()
    
    return edges
