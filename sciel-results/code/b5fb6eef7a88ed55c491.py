import numpy as np
import time
import itertools

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Precompute triangle indices
    tri_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_list.append((i, j, k))
    tri_arr = np.array(tri_list)
    I = tri_arr[:, 0]
    J = tri_arr[:, 1]
    K = tri_arr[:, 2]
    n_tri = len(tri_arr)
    
    # Precompute triangles incident to each vertex
    tri_incident = [[] for _ in range(n)]
    for t_idx in range(n_tri):
        u, v, w = tri_arr[t_idx]
        tri_incident[u].append(t_idx)
        tri_incident[v].append(t_idx)
        tri_incident[w].append(t_idx)
    tri_incident_arr = [np.array(lst, dtype=np.int32) for lst in tri_incident]

    # Initialize best_adj with K_{8,8} + edge (0,1)
    # Vertices 0-7 in A, 8-15 in B
    best_adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            best_adj[u, v] = 1
            best_adj[v, u] = 1
    # Add edge (0,1) within A
    best_adj[0, 1] = 1
    best_adj[1, 0] = 1

    def get_state(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0, np.zeros(n_tri, dtype=np.int32)
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        all_sums = np.zeros(n_tri, dtype=np.int32)
        all_sums[valid] = sums
        return int(np.max(all_sums)), all_sums

    best_score, best_sums = get_state(best_adj)
    current_adj = best_adj.copy()
    current_score, current_sums = best_score, best_sums.copy()
    current_deg = current_adj.sum(axis=1)

    def update_state(adj, sums, deg, u, v, x, y, added_xy, removed_uv):
        """
        Update degrees and triangle sums after swapping edge (u,v) with (x,y).
        If added_xy is True, (x,y) is added. If removed_uv is True, (u,v) is removed.
        Assumes adj is already updated.
        """
        # Update degrees
        if removed_uv:
            deg[u] -= 1
            deg[v] -= 1
        if added_xy:
            deg[x] += 1
            deg[y] += 1
            
        # Identify affected triangles
        # Union of triangles incident to u, v, x, y
        affected = np.unique(np.concatenate([tri_incident_arr[u], tri_incident_arr[v], tri_incident_arr[x], tri_incident_arr[y]]))
        
        # Recalculate sums for affected triangles
        for t_idx in affected:
            i, j, k = tri_arr[t_idx]
            if adj[i, j] and adj[j, k] and adj[k, i]:
                sums[t_idx] = deg[i] + deg[j] + deg[k]
            else:
                sums[t_idx] = 0
                
        new_score = int(np.max(sums))
        return new_score

    def get_edges_and_non_edges(adj):
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        return edges, non_edges

    # Local Search: Steepest Descent
    improved = True
    while improved and time.time() - start_time < deadline - 20:
        improved = False
        edges, non_edges = get_edges_and_non_edges(current_adj)
        
        # Find worst triangle to prioritize
        max_score = int(np.max(current_sums))
        worst_tris = np.where(current_sums == max_score)[0]
        worst_vertices = set()
        for t_idx in worst_tris:
            i, j, k = tri_arr[t_idx]
            worst_vertices.add(i)
            worst_vertices.add(j)
            worst_vertices.add(k)

        best_swap = None
        best_swap_score = current_score
        
        # Try all swaps
        # Optimization: Only try swaps that involve at least one vertex from worst triangles?
        # Or just all swaps? 120*119/2 is ~7000. 7000 * 16 updates is fast.
        for u, v in edges:
            # If (u,v) is not incident to a worst triangle, it might still help by reducing degrees elsewhere?
            # But breaking a worst triangle is most direct.
            # Let's stick to all swaps for completeness in the first pass, or filter if slow.
            
            for x, y in non_edges:
                if x == u or x == v or y == u or y == v:
                    continue
                
                # Apply swap
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = update_state(current_adj, current_sums, current_deg, u, v, x, y, True, True)
                
                # Revert
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_adj[x, y] = 0
                current_adj[y, x] = 0
                update_state(current_adj, current_sums, current_deg, u, v, x, y, False, False)
                
                if new_score < best_swap_score:
                    best_swap_score = new_score
                    best_swap = (u, v, x, y)
                    
                if time.time() - start_time > deadline - 20:
                    break
            if time.time() - start_time > deadline - 20:
                break

        if best_swap:
            u, v, x, y = best_swap
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            current_score = update_state(current_adj, current_sums, current_deg, u, v, x, y, True, True)
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj.copy()
                best_sums = current_sums.copy()
            improved = True

    # Simulated Annealing
    if time.time() - start_time < deadline - 5:
        T = 1.0
        cooling = 0.99
        # Use a fixed number of iterations or time-based
        steps = 0
        while time.time() - start_time < deadline - 5:
            edges, non_edges = get_edges_and_non_edges(current_adj)
            if not edges or not non_edges:
                break
                
            u, v = edges[np.random.randint(len(edges))]
            x, y = non_edges[np.random.randint(len(non_edges))]
            if x == u or x == v or y == u or y == v:
                continue
            
            # Apply swap
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            
            new_score = update_state(current_adj, current_sums, current_deg, u, v, x, y, True, True)
            
            delta = new_score - current_score
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
                    best_sums = current_sums.copy()
            else:
                # Revert
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_adj[x, y] = 0
                current_adj[y, x] = 0
                update_state(current_adj, current_sums, current_deg, u, v, x, y, False, False)
                
            T *= cooling
            if T < 0.1:
                T = 0.1
            steps += 1

    # Final Polish: One more pass of steepest descent
    if time.time() - start_time < deadline - 2:
        improved = True
        while improved and time.time() - start_time < deadline - 2:
            improved = False
            edges, non_edges = get_edges_and_non_edges(current_adj)
            best_swap = None
            best_swap_score = current_score
            
            for u, v in edges:
                for x, y in non_edges:
                    if x == u or x == v or y == u or y == v:
                        continue
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0
                    current_adj[x, y] = 1
                    current_adj[y, x] = 1
                    new_score = update_state(current_adj, current_sums, current_deg, u, v, x, y, True, True)
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
                    update_state(current_adj, current_sums, current_deg, u, v, x, y, False, False)
                    
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_swap = (u, v, x, y)
                        
            if best_swap:
                u, v, x, y = best_swap
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                current_score = update_state(current_adj, current_sums, current_deg, u, v, x, y, True, True)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    best_sums = current_sums.copy()
                improved = True

    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
    
    return {'n': n, 'edges': edges_list}
