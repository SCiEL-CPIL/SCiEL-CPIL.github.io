import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # 10s margin before 120s limit
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute all 4-cycles for fast validation
    all_4cycles = []
    for u in range(n):
        for v in range(u+1, n):
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    # 4-cycle: u-v-x-y-u (need all 4 edges to exist)
                    all_4cycles.append((u, v, x, y))
    all_4cycles = np.array(all_4cycles, dtype=np.int32)
    
    # Precompute witness indices for vectorization
    witness_data = {}
    for u in range(n):
        for v in range(u+1, n):
            vs = []
            xs = []
            ys = []
            us = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    vs.append(v)
                    xs.append(x)
                    ys.append(y)
                    us.append(u)
            
            witness_data[(u, v)] = (np.array(vs), np.array(xs), np.array(ys), np.array(us))
            witness_data[(v, u)] = witness_data[(u, v)]

    def can_add_edge_fast(adj, u, v, c):
        if u > v:
            u, v = v, u
        vs, xs, ys, us = witness_data[(u, v)]
        if len(vs) == 0:
            return True
            
        c_vx = adj[vs, xs]
        c_xy = adj[xs, ys]
        c_yu = adj[ys, us]
        
        # Check existence
        valid_mask = (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        if not np.any(valid_mask):
            return True
            
        # Check rainbow condition
        conflict = (
            (c_vx == c) | (c_xy == c) | (c_yu == c) |
            (c_vx == c_xy) | (c_vx == c_yu) | (c_xy == c_yu)
        )
        
        # If any valid witness has a conflict, it's invalid
        if np.any(valid_mask & conflict):
            return False
            
        return True
    
    def count_invalid_cycles(adj):
        """Count number of 4-cycles that violate the rainbow condition."""
        if len(all_4cycles) == 0:
            return 0
        u, v, x, y = all_4cycles[:, 0], all_4cycles[:, 1], all_4cycles[:, 2], all_4cycles[:, 3]
        c_uv = adj[u, v]
        c_vx = adj[v, x]
        c_xy = adj[x, y]
        c_yu = adj[y, u]
        
        # All four edges must exist
        valid = (c_uv != -1) & (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        if not np.any(valid):
            return 0
        
        # Rainbow: all 4 colors distinct
        rainbow = (
            (c_uv != c_vx) & (c_uv != c_xy) & (c_uv != c_yu) &
            (c_vx != c_xy) & (c_vx != c_yu) &
            (c_xy != c_yu)
        )
        
        return int(np.sum(valid & ~rainbow))
    
    def enforce_validity(adj):
        """Remove edges until the graph is valid. Returns number of edges removed."""
        removed = 0
        while time.time() < deadline:
            invalid_count = count_invalid_cycles(adj)
            if invalid_count == 0:
                break
            
            # Find an invalid cycle and remove one of its edges
            u, v, x, y = all_4cycles[:, 0], all_4cycles[:, 1], all_4cycles[:, 2], all_4cycles[:, 3]
            c_uv = adj[u, v]
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            valid = (c_uv != -1) & (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
            rainbow = (
                (c_uv != c_vx) & (c_uv != c_xy) & (c_uv != c_yu) &
                (c_vx != c_xy) & (c_vx != c_yu) &
                (c_xy != c_yu)
            )
            
            invalid_mask = valid & ~rainbow
            if not np.any(invalid_mask):
                break
            
            # Pick a random invalid cycle
            idx = np.random.choice(np.where(invalid_mask)[0])
            ui, vi, xi, yi = int(u[idx]), int(v[idx]), int(x[idx]), int(y[idx])
            
            # Remove one of the 4 edges (randomly)
            edges_to_consider = [(ui, vi), (vi, xi), (xi, yi), (yi, ui)]
            random.shuffle(edges_to_consider)
            for a, b in edges_to_consider:
                if adj[a, b] != -1:
                    adj[a, b] = -1
                    adj[b, a] = -1
                    removed += 1
                    break
        
        return removed

    def can_recolor_edge_fast(adj, u, v, new_c):
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        
        # Temporarily set
        adj[u, v] = new_c
        adj[v, u] = new_c
        
        valid = can_add_edge_fast(adj, u, v, new_c)
        
        # Revert
        adj[u, v] = old_c
        adj[v, u] = old_c
        
        return valid

    # Try multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Start with a random tree
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Try to add more edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            added = False
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                if can_add_edge_fast(adj, u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    added = True
                    break
            
            if not added:
                # Try to recolor an edge to make room
                # Try recoloring edges incident to u or v
                for neighbor in [u, v]:
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
                            continue
                        old_c = adj[neighbor, x]
                        for new_c in range(n):
                            if new_c == old_c:
                                continue
                            if can_recolor_edge_fast(adj, neighbor, x, new_c):
                                adj[neighbor, x] = new_c
                                adj[x, neighbor] = new_c
                                
                                # Try to add (u,v) again
                                added_after = False
                                for c2 in range(n):
                                    if can_add_edge_fast(adj, u, v, c2):
                                        adj[u, v] = c2
                                        adj[v, u] = c2
                                        added_after = True
                                        break
                                
                                if added_after:
                                    added = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        
                        if added:
                            break
                    if added:
                        break
        
        # Local Search Phase: Try to add edges in random order multiple times
        for _ in range(8):
            if time.time() > deadline: break
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline: break
                if adj[u, v] != -1: continue
                added = False
                colors = list(range(n))
                random.shuffle(colors)
                for c in colors:
                    if can_add_edge_fast(adj, u, v, c):
                        adj[u, v] = c
                        adj[v, u] = c
                        added = True
                        break
            
            # Recoloring pass: try to recolor existing edges to free up colors
            if time.time() > deadline: break
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] != -1]
            random.shuffle(existing)
            for u, v in existing[:50]:  # limit work
                if time.time() > deadline: break
                old_c = adj[u, v]
                # Try recoloring this edge
                new_colors = [c for c in range(n) if c != old_c]
                random.shuffle(new_colors)
                for new_c in new_colors[:5]:
                    if can_recolor_edge_fast(adj, u, v, new_c):
                        adj[u, v] = new_c
                        adj[v, u] = new_c
                        break
        
        # Enforce validity
        enforce_validity(adj)
        
        # Final polish: try to add back edges after removals
        for _ in range(3):
            if time.time() > deadline: break
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline: break
                if adj[u, v] != -1: continue
                colors = list(range(n))
                random.shuffle(colors)
                for c in colors:
                    if can_add_edge_fast(adj, u, v, c):
                        adj[u, v] = c
                        adj[v, u] = c
                        break
        
        cnt = np.sum(adj != -1) // 2
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 160:
            break
    
    # Final validity check and cleanup on best_adj
    if best_adj is not None:
        # Ensure no invalid cycles remain
        enforce_validity(best_adj)
        # Try one more polish pass
        for _ in range(2):
            if time.time() > deadline: break
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i, j] == -1]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline: break
                if best_adj[u, v] != -1: continue
                colors = list(range(n))
                random.shuffle(colors)
                for c in colors:
                    if can_add_edge_fast(best_adj, u, v, c):
                        best_adj[u, v] = c
                        best_adj[v, u] = c
                        break
    
    # Convert to result format
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
