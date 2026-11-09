import numpy as np
import time
import random

def construct():
    n = 31
    deadline = time.time() + 110  # Safety margin
    
    # 1. Generate polarity graph P
    vertices = []
    for x in range(5):
        for y in range(5):
            vertices.append((x, y, 1))
    for x in range(5):
        vertices.append((x, 1, 0))
    vertices.append((1, 0, 0))
    
    def are_adjacent(v1, v2):
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        return (x1 * x2 + y1 * y2 - z1 * z2) % 5 == 0

    # Build adjacency bitmasks for P
    adj_masks = [0] * n
    edges = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            if are_adjacent(vertices[i], vertices[j]):
                edges.add((i, j))
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                
    # 2. Find best edge to add to P
    # We want to minimize C4s created.
    # C4s created by adding uv is number of paths u-w-x-v in P.
    best_edge = None
    min_c4 = float('inf')
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edges:
                continue
            # Count paths i-w-j in P? No, i-w-x-j.
            # w in N(i), x in N(j), w ~ x, w != x.
            count = 0
            # Iterate w in N(i)
            mask_i = adj_masks[i]
            # Get list of neighbors
            neighbors_i = [k for k in range(n) if (mask_i >> k) & 1]
            for w in neighbors_i:
                if w == j: continue
                # x in N(j)
                mask_j = adj_masks[j]
                # x must be neighbor of w
                # So x in N(j) AND N(w)
                common = mask_j & adj_masks[w]
                # x != w, x != i, x != j
                # x != j is true since j not in N(j)
                # x != i: i is in N(j)? No, (i,j) not in edges. So i not in N(j).
                # x != w: w is in N(w)? No.
                # So just popcount(common)
                count += bin(common).count('1')
            
            if count < min_c4:
                min_c4 = count
                best_edge = (i, j)
                
    # Add best edge
    u, v = best_edge
    edges.add((u, v))
    adj_masks[u] |= (1 << v)
    adj_masks[v] |= (1 << u)
    
    # 3. Local Search
    # State: adj_masks, edges
    # Objective: Minimize total C4s
    
    def count_c4s(masks):
        total = 0
        # Count C4s containing each edge?
        # Or just iterate all 4-tuples?
        # n=31, n^4 is ~ 1M.
        # Better: For each pair of edges (u,v) and (x,y) disjoint?
        # Standard: Sum over pairs of vertices (u,v) of (common_neighbors(u,v) choose 2)?
        # No, that counts C4s where u,v are opposite?
        # Number of C4s = 1/4 * Sum_{u,v} ( |N(u) \cap N(v)| choose 2 )?
        # No, each C4 has 2 pairs of opposite vertices.
        # So Sum_{u < v} ( |N(u) \cap N(v)| choose 2 ) / 2 ?
        # Let's check.
        # C4: a-b-c-d-a.
        # Pairs of opposite: (a,c) and (b,d).
        # N(a) \cap N(c) contains b, d. Size >= 2.
        # N(b) \cap N(d) contains a, c. Size >= 2.
        # So each C4 is counted twice in the sum over pairs.
        # So Total C4s = 0.5 * Sum_{u < v} ( |N(u) \cap N(v)| choose 2 ).
        
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = masks[i] & masks[j]
                k = bin(common).count('1')
                if k >= 2:
                    s += k * (k - 1) // 2
        return s // 2

    current_c4s = count_c4s(adj_masks)
    best_c4s = current_c4s
    best_masks = adj_masks[:]
    best_edges = set(edges)
    
    # Local search loop
    while time.time() < deadline:
        # Pick random edge to remove
        edge_list = list(best_edges)
        if not edge_list:
            break
        rem_u, rem_v = random.choice(edge_list)
        
        # Pick random non-edge to add
        # Sample a few times to find a non-edge
        add_u = None
        add_v = None
        for _ in range(10):
            pu = random.randint(0, n - 1)
            pv = random.randint(0, n - 1)
            if pu == pv: continue
            if (pu, pv) in best_edges: continue
            if (pv, pu) in best_edges: continue
            add_u = pu
            add_v = pv
            break
            
        if add_u is None:
            continue
            
        # Ensure we don't remove the edge we are adding (impossible)
        # Ensure rem_u, rem_v != add_u, add_v?
        # If we remove (a,b) and add (a,b), no change.
        if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
            continue
            
        # Calculate new C4 count
        # Remove rem_u, rem_v
        new_masks = best_masks[:]
        new_masks[rem_u] &= ~(1 << rem_v)
        new_masks[rem_v] &= ~(1 << rem_u)
        
        # Add add_u, add_v
        new_masks[add_u] |= (1 << add_v)
        new_masks[add_v] |= (1 << add_u)
        
        new_c4s = count_c4s(new_masks)
        
        if new_c4s < best_c4s:
            best_c4s = new_c4s
            best_masks = new_masks
            # Update best_edges
            best_edges.remove((rem_u, rem_v))
            best_edges.add((add_u, add_v))
            # Normalize edge order? Not necessary for set, but for return
            # We store tuples.
            
    # Return list of edges
    # Ensure format [u, v]
    res = []
    for u, v in best_edges:
        res.append([u, v])
        
    return res
