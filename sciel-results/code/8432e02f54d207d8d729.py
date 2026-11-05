import numpy as np
import random
import math
import time

def construct():
    n = 20
    
    # Precompute triples
    triples = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                idx += 1
                
    num_triples = len(triples)
    
    # Precompute for each vertex u, the list of (t_idx, x, y) 
    # where t_idx is the index of triple (u, x, y)
    # This avoids looking up 'other verts' in a dict during the loop
    vertex_triples = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(triples):
        vertex_triples[i].append((t_idx, j, k))
        vertex_triples[j].append((t_idx, i, k))
        vertex_triples[k].append((t_idx, i, j))

    # Precompute penalty values for counts 0..n
    # Penalty for count c is (c-2)^2 if c > 2 else 0
    pen_vals = [0] * (n + 1)
    for c in range(n + 1):
        if c > 2:
            pen_vals[c] = (c - 2) ** 2
        else:
            pen_vals[c] = 0

    # Initialize random graph
    adj = np.zeros((n, n), dtype=np.uint8)
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                adj[u, v] = 1
                adj[v, u] = 1
                
    edge_count = np.sum(adj) // 2
    
    # Initialize counts and penalty
    counts = np.zeros(num_triples, dtype=np.int32)
    penalty = 0
    
    for t_idx, (i, j, k) in enumerate(triples):
        c = 0
        for m in range(n):
            if adj[i, m] and adj[j, m] and adj[k, m]:
                c += 1
        counts[t_idx] = c
        penalty += pen_vals[c]
        
    # SA Parameters
    start_time = time.time()
    timeout = 115.0
    
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    Lambda = 100.0
    
    best_edges = 0
    best_adj = None
    
    if penalty == 0:
        best_edges = edge_count
        best_adj = adj.copy()
        
    while time.time() - start_time < timeout and T > min_T:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        is_add = not adj[u, v]
        
        delta_pen = 0
        
        # Update for triples containing u
        for t_idx, x, y in vertex_triples[u]:
            # Check if v is in the triple (u, x, y)
            # The triple is (u, x, y). v is in it if v==x or v==y.
            # If v is in the triple, the edge (u,v) is part of the triple's definition?
            # No, the triple is a set of 3 vertices. The edge (u,v) connects two of them.
            # If v == x or v == y, then the triple contains both u and v.
            # In that case, flipping (u,v) does NOT change the common neighbor count for that triple?
            # Let's check: C_ijk = sum_m x_im x_jm x_km.
            # If triple is (u, v, w), C = sum_m x_um x_vm x_wm.
            # Flipping x_uv changes x_um for m=v and x_vm for m=u.
            # Term for m=v: x_uv x_vv(0) x_wv = 0.
            # Term for m=u: x_uu(0) x_vu x_wu = 0.
            # So if u and v are in the same triple, flipping their edge does NOT affect the count for that triple.
            # Therefore, we only care about triples containing u but NOT v.
            if v == x or v == y:
                continue
                
            if is_add:
                if adj[x, v] and adj[y, v]:
                    old_c = counts[t_idx]
                    new_c = old_c + 1
                    delta_pen += pen_vals[new_c] - pen_vals[old_c]
                    counts[t_idx] = new_c
            else:
                if adj[x, v] and adj[y, v]:
                    old_c = counts[t_idx]
                    new_c = old_c - 1
                    delta_pen += pen_vals[new_c] - pen_vals[old_c]
                    counts[t_idx] = new_c
                    
        # Update for triples containing v
        for t_idx, x, y in vertex_triples[v]:
            if u == x or u == y:
                continue
                
            if is_add:
                if adj[x, u] and adj[y, u]:
                    old_c = counts[t_idx]
                    new_c = old_c + 1
                    delta_pen += pen_vals[new_c] - pen_vals[old_c]
                    counts[t_idx] = new_c
            else:
                if adj[x, u] and adj[y, u]:
                    old_c = counts[t_idx]
                    new_c = old_c - 1
                    delta_pen += pen_vals[new_c] - pen_vals[old_c]
                    counts[t_idx] = new_c
                    
        delta_edges = 1 if is_add else -1
        delta_E = -delta_edges + Lambda * delta_pen
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            adj[u, v] = 1 if is_add else 0
            adj[v, u] = 1 if is_add else 0
            edge_count += delta_edges
            penalty += delta_pen
            
            if penalty == 0 and edge_count > best_edges:
                best_edges = edge_count
                best_adj = adj.copy()
        
        T *= cooling
        
    if best_adj is None:
        # Fallback: Try to find any feasible graph
        # This is unlikely with Lambda=100
        return []
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return edges
