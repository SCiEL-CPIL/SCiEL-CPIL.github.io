import time
import random
import numpy as np
from itertools import combinations

def construct():
    n = 31
    m = 91
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Greedy construction: add edges one by one, choosing the edge that
    # creates the fewest new 4-cycles (i.e., has fewest common neighbors)
    adj = np.zeros((n, n), dtype=np.int32)
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_edges)
    
    for _ in range(m):
        if time.time() > deadline - 30:
            break
        best_edge = None
        best_common = float('inf')
        best_deg_diff = float('inf')
        
        # Sample a subset of candidate edges for speed
        sample_size = min(len(all_edges), 500)
        candidates = random.sample(all_edges, sample_size) if len(all_edges) > sample_size else all_edges
        
        for u, v in candidates:
            if adj[u, v]:
                continue
            common = np.sum(adj[u] & adj[v])
            deg_diff = abs(np.sum(adj[u]) - np.sum(adj[v]))
            if common < best_common or (common == best_common and deg_diff < best_deg_diff):
                best_common = common
                best_deg_diff = deg_diff
                best_edge = (u, v)
        
        if best_edge is None:
            # Fallback: pick any unadded edge
            for u, v in all_edges:
                if not adj[u, v]:
                    best_edge = (u, v)
                    break
        
        if best_edge is None:
            break
            
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append([u, v])
        all_edges.remove(best_edge)
    
    # Local search: try edge swaps to reduce 4-cycles
    def count_c4(adj):
        """Count 4-cycles in the graph."""
        c4_count = 0
        # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da
        # Count pairs of vertices with >= 2 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count
    
    current_c4 = count_c4(adj)
    
    # Build edge set for quick lookup
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    # Local search with edge swaps
    max_swaps = 2000
    no_improve_count = 0
    max_no_improve = 100
    
    for iteration in range(max_swaps):
        if time.time() > deadline - 15:
            break
        if no_improve_count > max_no_improve:
            break
        
        # Pick a random existing edge and a random non-edge
        if len(edges) < 2:
            break
            
        u1, v1 = random.choice(edges)
        # Find a non-edge
        attempts = 0
        while attempts < 50:
            u2, v2 = random.randint(0, n-1), random.randint(0, n-1)
            if u2 == v2:
                attempts += 1
                continue
            key = (min(u2,v2), max(u2,v2))
            if key not in edge_set:
                break
            attempts += 1
        else:
            continue
        
        # Check if swap is valid (no duplicate edges)
        key1 = (min(u1,v1), max(u1,v1))
        key2 = (min(u2,v2), max(u2,v2))
        
        # Try the swap: remove (u1,v1), add (u2,v2)
        # Calculate change in 4-cycles
        # Removing edge (u1,v1): reduces 4-cycles by common(u1,v1)
        # Adding edge (u2,v2): increases 4-cycles by common(u2,v2)
        common_remove = np.sum(adj[u1] & adj[v1])
        common_add = np.sum(adj[u2] & adj[v2])
        
        delta = common_add - common_remove
        
        if delta < 0:
            # Accept the swap
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            edges[edges.index([u1, v1])] = [u2, v2]
            edge_set.discard(key1)
            edge_set.add(key2)
            current_c4 += delta
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Verify we have exactly m edges
    result = [[min(u,v), max(u,v)] for u, v in edges]
    result = list(set((tuple(e) for e in result)))
    result = [list(e) for e in result]
    
    if len(result) != m:
        # Fallback to greedy result
        result = [[min(u,v), max(u,v)] for u, v in edges[:m]]
    
    return result[:m]
