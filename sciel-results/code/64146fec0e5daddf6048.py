import time
import random
import numpy as np

def construct():
    n = 16
    # Initialize with seed
    # K_{8,8} + (0,1)
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    # Precompute bitmasks for adjacency
    def build_masks(adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for i in range(n):
                if adj[u, i]:
                    m |= (1 << i)
            masks[u] = m
        return masks
    
    def get_score_fast(adj, masks):
        deg = adj.sum(axis=1)
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common_mask &= common_mask - 1
        return max_sum
    
    current_score = get_score_fast(adj, build_masks(adj))
    best_adj = adj.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_possible_set = set(all_possible_edges)
    
    # Simulated annealing
    current_adj = adj.copy()
    current_masks = build_masks(current_adj)
    current_deg = current_adj.sum(axis=1)
    current_score = get_score_fast(current_adj, current_masks)
    
    best_adj = current_adj.copy()
    best_score = current_score
    best_masks = current_masks.copy()
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    total_iters = 50000
    
    for it in range(total_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Update temperature
        frac = it / total_iters
        T = T_start * (T_end / T_start) ** frac
        
        # Choose a move: remove one edge, add one non-edge
        # Preferentially remove edges that are in high-degree-sum triangles
        # and add edges that break those triangles or reduce degree concentration
        
        # Get current edges as set for fast lookup
        cur_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u, v]:
                    cur_edges.add((u, v))
        
        # Find worst triangle and its edges
        worst_tri_edges = []
        worst_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u, v]:
                    cm = current_masks[u] & current_masks[v]
                    while cm:
                        lsb = cm & -cm
                        w = lsb.bit_length() - 1
                        s = current_deg[u] + current_deg[v] + current_deg[w]
                        if s > worst_sum:
                            worst_sum = s
                            worst_tri_edges = [(u, v), (u, w), (v, w)]
                        cm &= cm - 1
        
        # With some probability, remove an edge from the worst triangle
        if worst_tri_edges and random.random() < 0.7:
            u, v = random.choice(worst_tri_edges)
        else:
            # Random edge
            u, v = random.choice(list(cur_edges))
        
        # Choose non-edge to add
        # Prefer non-edges that would break the worst triangle or reduce degree imbalance
        non_edges = [e for e in all_possible_edges if e not in cur_edges]
        
        if not non_edges:
            continue
        
        # Score potential additions: prefer edges that connect low-degree vertices
        # or break triangles
        if worst_tri_edges and random.random() < 0.5:
            # Try to add an edge that's not in the worst triangle's vertices
            tri_verts = set()
            for a, b in worst_tri_edges:
                tri_verts.add(a)
                tri_verts.add(b)
            # Prefer non-edges outside the triangle
            outside = [e for e in non_edges if e[0] not in tri_verts and e[1] not in tri_verts]
            if outside:
                x, y = random.choice(outside)
            else:
                x, y = random.choice(non_edges)
        else:
            x, y = random.choice(non_edges)
        
        # Apply move
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1
        
        # Update masks
        current_masks[u] &= ~(1 << v)
        current_masks[v] &= ~(1 << u)
        current_masks[x] |= (1 << y)
        current_masks[y] |= (1 << x)
        
        # Update degrees
        current_deg[u] -= 1
        current_deg[v] -= 1
        current_deg[x] += 1
        current_deg[y] += 1
        
        new_score = get_score_fast(current_adj, current_masks)
        
        # Accept if better or with probability exp(-(new-old)/T)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
                best_masks = current_masks.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_adj[x, y] = 0
                current_adj[y, x] = 0
                current_masks[u] |= (1 << v)
                current_masks[v] |= (1 << u)
                current_masks[x] &= ~(1 << y)
                current_masks[y] &= ~(1 << x)
                current_deg[u] += 1
                current_deg[v] += 1
                current_deg[x] -= 1
                current_deg[y] -= 1
    
    # Convert best_adj to edges list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
