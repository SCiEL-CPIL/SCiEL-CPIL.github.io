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

    def try_polarity_subset(n_target, q=7):
        adj_full, n_full = get_polarity_graph(q)
        if n_full < n_target:
            return None
        target_deg = n_target - 39
        num_remove = n_full - n_target
        degs_full = adj_full.sum(axis=1)
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            R = set()
            counts = np.zeros(n_full, dtype=np.int8)
            order = list(range(n_full))
            random.shuffle(order)
            removed = 0
            for v in order:
                if removed == num_remove:
                    break
                can_remove = True
                for u in range(n_full):
                    if adj_full[v][u] == 1:
                        if u not in R:
                            if counts[u] >= degs_full[u] - target_deg:
                                can_remove = False
                                break
                if can_remove:
                    R.add(v)
                    removed += 1
                    for u in range(n_full):
                        if adj_full[v][u] == 1:
                            counts[u] += 1
            if removed == num_remove:
                S = [i for i in range(n_full) if i not in R]
                if len(S) != n_target:
                    continue
                adj_sub = np.zeros((n_target, n_target), dtype=np.int8)
                for new_i, old_i in enumerate(S):
                    for new_j, old_j in enumerate(S):
                        if new_i < new_j:
                            if adj_full[old_i][old_j]:
                                adj_sub[new_i][new_j] = 1
                                adj_sub[new_j][new_i] = 1
                degs_sub = adj_sub.sum(axis=1)
                if np.min(degs_sub) >= target_deg:
                    edges = []
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            if adj_sub[i][j]:
                                edges.append([i, j])
                    return {'n': n_target, 'edges': edges}
        return None

    def er_local_search(n, target_deg, max_iters=40000, p=0.2):
        """ER random graph init + local search: remove C4 edges, add edges to low-degree vertices."""
        adj = np.zeros((n, n), dtype=np.int8)
        degs = np.zeros(n, dtype=np.int8)
        
        # Initialize with ER random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    degs[i] += 1
                    degs[j] += 1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if np.min(degs) < target_deg:
                # Need to add edges to low-degree vertices
                low_verts = list(np.where(degs < target_deg)[0])
                random.shuffle(low_verts)
                added = False
                for v in low_verts:
                    non_neighbors = [u for u in range(n) if u != v and adj[v][u] == 0]
                    random.shuffle(non_neighbors)
                    for u in non_neighbors:
                        # Check if adding (u,v) creates C4
                        safe = True
                        Nu = np.where(adj[u] == 1)[0]
                        for w in Nu:
                            if np.any(adj[v][np.where(adj[w] == 1)[0]]):
                                safe = False
                                break
                        if safe:
                            adj[u][v] = 1
                            adj[v][u] = 1
                            degs[u] += 1
                            degs[v] += 1
                            added = True
                            break
                    if added:
                        break
                if not added:
                    break
            else:
                # Min degree OK, check for C4s and remove one edge
                found_c4 = False
                for u in range(n):
                    if found_c4:
                        break
                    for v in range(u + 1, n):
                        if adj[u][v] == 0:
                            continue
                        common = np.sum(adj[u] & adj[v])
                        if common >= 2:
                            adj[u][v] = 0
                            adj[v][u] = 0
                            degs[u] -= 1
                            degs[v] -= 1
                            found_c4 = True
                            break
                if not found_c4:
                    # No C4 and min degree OK: valid solution
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return edges
        
        return None

    # Try n=47, 46, 45...
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        target_deg = n - 39
        if target_deg <= 0:
            return {'n': n, 'edges': []}
        
        # Try Polarity Graph Subset first
        if n <= 57:
            res = try_polarity_subset(n)
            if res:
                return res
        
        # Try ER + Local Search
        for attempt in range(15):
            if time.time() > deadline:
                break
            edges = er_local_search(n, target_deg, max_iters=40000)
            if edges is not None:
                return {'n': n, 'edges': edges}
        
        # Fallback: Random Greedy Construction
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for attempt in range(30):
            if time.time() > deadline:
                break
            random.shuffle(all_edges)
            adj = np.zeros((n, n), dtype=np.int8)
            degs = np.zeros(n, dtype=np.int8)
            for u, v in all_edges:
                N_u = np.where(adj[u] == 1)[0]
                valid = True
                for w in N_u:
                    if np.any(adj[w] & adj[v]):
                        valid = False
                        break
                if valid:
                    adj[u][v] = 1
                    adj[v][u] = 1
                    degs[u] += 1
                    degs[v] += 1
            if np.min(degs) >= target_deg:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Trivial fallback
    return {'n': 39, 'edges': []}
