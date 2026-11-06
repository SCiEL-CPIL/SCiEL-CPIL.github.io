import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def get_polarity_graph(q):
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, -1, q)
                        points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                    elif y != 0:
                        inv_y = pow(y, -1, q)
                        points.append((0, 1, (z * inv_y) % q))
                    else:
                        inv_z = pow(z, -1, q)
                        points.append((0, 0, 1))
        unique_points = list(set(points))
        unique_points.sort()
        n_pts = len(unique_points)
        adj = np.zeros((n_pts, n_pts), dtype=np.int8)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                u = unique_points[i]
                v = unique_points[j]
                dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
                if dot == 0:
                    adj[i][j] = 1
                    adj[j][i] = 1
        return adj, n_pts

    def can_add_edge(adj, u, v):
        # Check if adding (u, v) creates a C4
        # C4 would be u - w - x - v - u
        # So we need to check if there exists w in N(u) and x in N(v) such that adj[w][x] == 1
        Nu = np.where(adj[u] == 1)[0]
        Nv = np.where(adj[v] == 1)[0]
        for w in Nu:
            for x in Nv:
                if adj[w][x] == 1:
                    return False
        return True

    def try_polarity_subset_with_additions(n_target, q=7):
        adj_full, n_full = get_polarity_graph(q)
        if n_full < n_target:
            return None
        target_deg = n_target - 39
        num_remove = n_full - n_target
        
        # We want to remove num_remove vertices
        # Then add edges to restore min degree to target_deg
        
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            # Randomly select vertices to remove
            # We should prioritize removing high-degree vertices? Or random?
            # Random is fine, but maybe we should avoid removing vertices that are critical?
            # Let's just shuffle and pick
            order = list(range(n_full))
            random.shuffle(order)
            R = set(order[:num_remove])
            S = [i for i in range(n_full) if i not in R]
            
            # Build induced subgraph
            adj_sub = np.zeros((n_target, n_target), dtype=np.int8)
            # Map old indices to new indices
            idx_map = {old: new for new, old in enumerate(S)}
            
            for new_i, old_i in enumerate(S):
                for new_j in range(new_i + 1, n_target):
                    old_j = S[new_j]
                    if adj_full[old_i][old_j]:
                        adj_sub[new_i][new_j] = 1
                        adj_sub[new_j][new_i] = 1
                        
            degs = adj_sub.sum(axis=1)
            
            # Add edges to boost min degree
            success = True
            while np.min(degs) < target_deg:
                # Find vertex with min degree
                v = int(np.argmin(degs))
                if degs[v] >= target_deg:
                    break
                    
                # Find a non-neighbor u to add edge to
                non_neighbors = [u for u in range(n_target) if u != v and adj_sub[v][u] == 0]
                random.shuffle(non_neighbors)
                
                added = False
                for u in non_neighbors:
                    if can_add_edge(adj_sub, v, u):
                        # Add edge
                        adj_sub[v][u] = 1
                        adj_sub[u][v] = 1
                        degs[v] += 1
                        degs[u] += 1
                        added = True
                        break
                        
                if not added:
                    success = False
                    break
            
            if success and np.min(degs) >= target_deg:
                # Verify C4-free (just in case)
                # and min degree
                if np.min(degs) >= target_deg:
                    # Check C4
                    is_c4_free = True
                    for i in range(n_target):
                        if not is_c4_free: break
                        for j in range(i + 1, n_target):
                            if adj_sub[i][j]:
                                common = np.sum(adj_sub[i] & adj_sub[j])
                                if common > 1:
                                    is_c4_free = False
                                    break
                    if is_c4_free:
                        edges = []
                        for i in range(n_target):
                            for j in range(i + 1, n_target):
                                if adj_sub[i][j]:
                                    edges.append([i, j])
                        return {'n': n_target, 'edges': edges}
                        
        return None

    # Try n=47, 46, 45...
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        target_deg = n - 39
        if target_deg <= 0:
            return {'n': n, 'edges': []}
            
        # Try Polarity Graph with Additions
        res = try_polarity_subset_with_additions(n)
        if res:
            return res
            
    # Fallback to previous logic if all else fails (unlikely to be needed if 46 works)
    return {'n': 39, 'edges': []}
