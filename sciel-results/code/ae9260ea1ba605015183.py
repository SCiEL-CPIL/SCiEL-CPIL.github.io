import time
import random
import numpy as np

def construct():
    n = 16
    # Initialize with K_{8,8} + (0,1)
    # This has 64 + 1 = 65 edges.
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

    def get_score_fast(adj, masks, deg=None):
        if deg is None:
            deg = adj.sum(axis=1)
        max_sum = 0
        # Iterate over edges
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

    def find_worst_triangle_info(adj, masks, deg):
        """Return list of (sum, edge1, edge2, edge3) for all triangles, sorted by sum descending."""
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        triangles.append((s, (u, v), (u, w), (v, w)))
                        common_mask &= common_mask - 1
        triangles.sort(key=lambda x: -x[0])
        return triangles

    def compute_affected_triangles(adj, masks, deg, removed_edge, added_edge):
        """Compute score considering only triangles affected by the move."""
        u, v = removed_edge
        x, y = added_edge
        
        # Triangles that existed before but are destroyed by removing (u,v)
        # These had (u,v) as an edge, so they contain both u and v
        # After removal, they no longer exist
        # But we need to know what the max was among remaining triangles
        
        # Actually, let's just recompute the full score but with early termination
        # if we can prove it's bounded
        
        max_sum = 0
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b]:
                    common_mask = masks[a] & masks[b]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        c = lsb.bit_length() - 1
                        s = deg[a] + deg[b] + deg[c]
                        if s > max_sum:
                            max_sum = s
                        common_mask &= common_mask - 1
        return max_sum

    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit
    
    best_adj = adj.copy()
    best_masks = build_masks(best_adj)
    best_deg = best_adj.sum(axis=1)
    best_score = get_score_fast(best_adj, best_masks, best_deg)

    # Simulated Annealing / Local Search
    current_adj = adj.copy()
    current_masks = build_masks(current_adj)
    current_deg = current_adj.sum(axis=1)
    current_score = get_score_fast(current_adj, current_masks, current_deg)

    T_start = 5.0
    T_end = 0.1
    total_iters = 200000 # Will be cut by time limit
    
    # Precompute all possible edges and non-edges for faster sampling
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for it in range(total_iters):
        if time.time() - start_time > time_limit:
            break

        frac = it / total_iters
        T = T_start * (T_end / T_start) ** frac

        # 1. Select edge to remove
        # Use worst triangles to guide selection, but cache the triangle list
        # to avoid recomputing every iteration
        triangles = find_worst_triangle_info(current_adj, current_masks, current_deg)
        
        if triangles and random.random() < 0.85:
            # Pick from top 3 worst triangles
            top_k = min(3, len(triangles))
            worst_tri = random.choice(triangles[:top_k])
            u, v = random.choice([worst_tri[1], worst_tri[2], worst_tri[3]])
        else:
            # Pick random edge
            u, v = random.choice(all_edges)
            while not current_adj[u, v]:
                u, v = random.choice(all_edges)
        
        # 2. Select non-edge to add
        # Prefer non-edges connecting lower-degree vertices
        # Build list of non-edges with weights based on degree sum
        non_edges = []
        for a, b in all_edges:
            if not current_adj[a, b]:
                # Lower degree sum = better candidate
                weight = 1.0 / (1.0 + current_deg[a] + current_deg[b])
                non_edges.append((a, b, weight))
        
        if non_edges:
            # Weighted random selection
            total_w = sum(w for _, _, w in non_edges)
            r = random.random() * total_w
            cum = 0.0
            x, y = non_edges[0][0], non_edges[0][1]
            for a, b, w in non_edges:
                cum += w
                if r <= cum:
                    x, y = a, b
                    break
        else:
            # Graph is complete, shouldn't happen with 65 edges on 16 vertices
            break

        # 3. Apply move
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1

        current_masks[u] &= ~(1 << v)
        current_masks[v] &= ~(1 << u)
        current_masks[x] |= (1 << y)
        current_masks[y] |= (1 << x)

        current_deg[u] -= 1
        current_deg[v] -= 1
        current_deg[x] += 1
        current_deg[y] += 1

        new_score = get_score_fast(current_adj, current_masks, current_deg)

        # 4. Accept/Reject
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
                best_masks = current_masks.copy()
                best_deg = current_deg.copy()
        else:
            delta = new_score - current_score
            if delta == 0 or random.random() < np.exp(-delta / T):
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

    # Final polish: deterministic local search
    # Try all single-edge swaps and keep improvements
    polish_adj = best_adj.copy()
    polish_masks = build_masks(polish_adj)
    polish_deg = polish_adj.sum(axis=1)
    polish_score = get_score_fast(polish_adj, polish_masks, polish_deg)
    
    improved = True
    polish_deadline = start_time + time_limit - 5  # Leave 5s margin
    while improved and time.time() < polish_deadline:
        improved = False
        for u, v in all_edges:
            if time.time() > polish_deadline:
                break
            if not polish_adj[u, v]:
                continue
            for x, y in all_edges:
                if x == u or x == v or y == u or y == v:
                    continue
                if polish_adj[x, y]:
                    continue
                
                # Try swap: remove (u,v), add (x,y)
                polish_adj[u, v] = 0
                polish_adj[v, u] = 0
                polish_adj[x, y] = 1
                polish_adj[y, x] = 1

                polish_masks[u] &= ~(1 << v)
                polish_masks[v] &= ~(1 << u)
                polish_masks[x] |= (1 << y)
                polish_masks[y] |= (1 << x)

                polish_deg[u] -= 1
                polish_deg[v] -= 1
                polish_deg[x] += 1
                polish_deg[y] += 1

                new_polish_score = get_score_fast(polish_adj, polish_masks, polish_deg)

                if new_polish_score < polish_score:
                    polish_score = new_polish_score
                    improved = True
                else:
                    # Revert
                    polish_adj[u, v] = 1
                    polish_adj[v, u] = 1
                    polish_adj[x, y] = 0
                    polish_adj[y, x] = 0

                    polish_masks[u] |= (1 << v)
                    polish_masks[v] |= (1 << u)
                    polish_masks[x] &= ~(1 << y)
                    polish_masks[y] &= ~(1 << x)

                    polish_deg[u] += 1
                    polish_deg[v] += 1
                    polish_deg[x] -= 1
                    polish_deg[y] -= 1

    # Use the better of best_adj and polish_adj
    if polish_score < best_score:
        best_adj = polish_adj
        best_score = polish_score

    # Convert best_adj to edges list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
