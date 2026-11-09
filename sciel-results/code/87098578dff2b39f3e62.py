import time
import random
import numpy as np

def construct():
    n = 16
    # Initialize with K_{8,8} + (0,1)
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    def build_masks(adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for i in range(n):
                if adj[u, i]:
                    m |= (1 << i)
            masks[u] = m
        return masks

    def get_score_and_sq_sum(adj, masks, deg=None):
        if deg is None:
            deg = adj.sum(axis=1)
        max_sum = 0
        sq_sum = 0
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
                        sq_sum += s * s
                        common_mask &= common_mask - 1
        return max_sum, sq_sum

    def get_worst_triangles(adj, masks, deg, k=5):
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        triangles.append((s, u, v, w))
                        common_mask &= common_mask - 1
        triangles.sort(key=lambda x: -x[0])
        return triangles[:k]

    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    current_adj = adj.copy()
    current_masks = build_masks(current_adj)
    current_deg = current_adj.sum(axis=1)
    current_max, current_sq = get_score_and_sq_sum(current_adj, current_masks, current_deg)
    
    best_adj = current_adj.copy()
    best_masks = current_masks.copy()
    best_deg = current_deg.copy()
    best_max = current_max
    best_sq = current_sq

    # Simulated Annealing with targeted triangle-breaking moves
    T_start = 80.0
    T_end = 0.5
    total_iters = 400000

    for it in range(total_iters):
        if time.time() - start_time > time_limit:
            break

        frac = it / total_iters
        T = T_start * (T_end / T_start) ** frac

        # Move: remove an edge from the worst triangle, add a random non-edge
        worst_tris = get_worst_triangles(current_adj, current_masks, current_deg, k=3)
        
        # Pick edge to remove
        if worst_tris and random.random() < 0.8:
            tri = random.choice(worst_tris)
            _, u, v, w = tri
            edge_pair = random.choice([(u, v), (u, w), (v, w)])
            ru, rv = edge_pair
            if not current_adj[ru, rv]:
                ru, rv = random.choice(all_edges)
                while not current_adj[ru, rv]:
                    ru, rv = random.choice(all_edges)
        else:
            ru, rv = random.choice(all_edges)
            while not current_adj[ru, rv]:
                ru, rv = random.choice(all_edges)

        # Pick non-edge to add
        while True:
            rx, ry = random.choice(all_edges)
            if not current_adj[rx, ry]:
                break

        # Apply
        current_adj[ru, rv] = 0
        current_adj[rv, ru] = 0
        current_adj[rx, ry] = 1
        current_adj[ry, rx] = 1

        current_masks[ru] &= ~(1 << rv)
        current_masks[rv] &= ~(1 << ru)
        current_masks[rx] |= (1 << ry)
        current_masks[ry] |= (1 << rx)

        current_deg[ru] -= 1
        current_deg[rv] -= 1
        current_deg[rx] += 1
        current_deg[ry] += 1

        new_max, new_sq = get_score_and_sq_sum(current_adj, current_masks, current_deg)
        
        delta_max = new_max - current_max
        
        if delta_max < 0:
            accept = True
        elif delta_max == 0:
            if new_sq < current_sq:
                accept = True
            else:
                delta_sq = new_sq - current_sq
                accept = random.random() < np.exp(-delta_sq / (T * 8))
        else:
            accept = random.random() < np.exp(-delta_max / T)

        if accept:
            current_max = new_max
            current_sq = new_sq
            if new_max < best_max:
                best_max = new_max
                best_adj = current_adj.copy()
                best_masks = current_masks.copy()
                best_deg = current_deg.copy()
        else:
            current_adj[ru, rv] = 1
            current_adj[rv, ru] = 1
            current_adj[rx, ry] = 0
            current_adj[ry, rx] = 0

            current_masks[ru] |= (1 << rv)
            current_masks[rv] |= (1 << ru)
            current_masks[rx] &= ~(1 << ry)
            current_masks[ry] &= ~(1 << rx)

            current_deg[ru] += 1
            current_deg[rv] += 1
            current_deg[rx] -= 1
            current_deg[ry] -= 1

    # Deterministic polish: try every edge removal and best replacement
    polish_adj = best_adj.copy()
    polish_masks = build_masks(polish_adj)
    polish_deg = polish_adj.sum(axis=1)
    polish_max, polish_sq = get_score_and_sq_sum(polish_adj, polish_masks, polish_deg)
    
    improved = True
    polish_deadline = start_time + time_limit - 3
    
    while improved and time.time() < polish_deadline:
        improved = False
        for u, v in all_edges:
            if time.time() > polish_deadline:
                break
            if not polish_adj[u, v]:
                continue
            
            best_add = None
            best_new_max = polish_max
            best_new_sq = polish_sq
            
            # Try removing (u,v) and adding each possible non-edge
            polish_adj[u, v] = 0
            polish_adj[v, u] = 0
            polish_masks[u] &= ~(1 << v)
            polish_masks[v] &= ~(1 << u)
            polish_deg[u] -= 1
            polish_deg[v] -= 1
            
            for x, y in all_edges:
                if x == u or x == v or y == u or y == v:
                    continue
                if polish_adj[x, y]:
                    continue
                
                # Add (x, y)
                polish_adj[x, y] = 1
                polish_adj[y, x] = 1
                polish_masks[x] |= (1 << y)
                polish_masks[y] |= (1 << x)
                polish_deg[x] += 1
                polish_deg[y] += 1
                
                new_max, new_sq = get_score_and_sq_sum(polish_adj, polish_masks, polish_deg)
                
                # Remove (x, y) for next iteration
                polish_adj[x, y] = 0
                polish_adj[y, x] = 0
                polish_masks[x] &= ~(1 << y)
                polish_masks[y] &= ~(1 << x)
                polish_deg[x] -= 1
                polish_deg[y] -= 1
                
                if new_max < best_new_max or (new_max == best_new_max and new_sq < best_new_sq):
                    best_new_max = new_max
                    best_new_sq = new_sq
                    best_add = (x, y)
            
            # Restore (u, v)
            polish_adj[u, v] = 1
            polish_adj[v, u] = 1
            polish_masks[u] |= (1 << v)
            polish_masks[v] |= (1 << u)
            polish_deg[u] += 1
            polish_deg[v] += 1
            
            if best_add is not None and best_new_max < polish_max:
                rx, ry = best_add
                polish_adj[rx, ry] = 1
                polish_adj[ry, rx] = 1
                polish_masks[rx] |= (1 << ry)
                polish_masks[ry] |= (1 << rx)
                polish_deg[rx] += 1
                polish_deg[ry] += 1
                polish_adj[u, v] = 0
                polish_adj[v, u] = 0
                polish_masks[u] &= ~(1 << v)
                polish_masks[v] &= ~(1 << u)
                polish_deg[u] -= 1
                polish_deg[v] -= 1
                polish_max = best_new_max
                polish_sq = best_new_sq
                improved = True

    if polish_max < best_max:
        best_adj = polish_adj
        best_max = polish_max

    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
