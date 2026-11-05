import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Construct Polarity Graph of PG(2,7)
    q = 7
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_total = len(points) # 57
    
    def is_orthogonal(p, qpt):
        x1, y1, z1 = p
        x2, y2, z2 = qpt
        return (x1 * y2 + y1 * x2 + z1 * z2) % q == 0

    # We want to select a subset of vertices of size up to 43.
    # Since the whole graph is C4-free and alpha <= 8, any induced subgraph is valid.
    # We can just pick the first 43 vertices.
    target_n = 43
    if target_n > n_total:
        target_n = n_total
        
    selected_indices = list(range(target_n))
    
    # Build adjacency list for the induced subgraph
    # Map old index to new index
    idx_map = {old: new for new, old in enumerate(selected_indices)}
    
    adj = [set() for _ in range(target_n)]
    
    for i_old in selected_indices:
        i_new = idx_map[i_old]
        for j_old in selected_indices:
            if j_old > i_old:
                j_new = idx_map[j_old]
                if is_orthogonal(points[i_old], points[j_old]):
                    adj[i_new].add(j_new)
                    adj[j_new].add(i_new)
                    
    # Construct edges list
    edges = []
    for u in range(target_n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': target_n, 'edges': edges}
