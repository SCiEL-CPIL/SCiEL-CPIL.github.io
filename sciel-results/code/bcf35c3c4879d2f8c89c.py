import numpy as np
import time
import itertools

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

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

    # Triangles incident to each vertex
    tri_incident = [[] for _ in range(n)]
    for t_idx in range(n_tri):
        u, v, w = tri_arr[t_idx]
        tri_incident[u].append(t_idx)
        tri_incident[v].append(t_idx)
        tri_incident[w].append(t_idx)
    tri_incident_arr = [np.array(lst) for lst in tri_incident]

    def get_score_full(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return int(np.max(sums))

    def compute_sums(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        all_sums = np.zeros(n_tri, dtype=np.int32)
        all_sums[valid] = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return all_sums

    def apply_swap(adj, sums, u, v, x, y):
        """Apply swap: remove (u,v), add (x,y). Update sums for all affected triangles."""
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        deg = adj.sum(axis=1)
        # Affected triangles: those containing u, v, x, or y
        affected = np.unique(np.concatenate([
            tri_incident_arr[u], tri_incident_arr[v],
            tri_incident_arr[x], tri_incident_arr[y]
        ]))
        for t_idx in affected:
            i, j, k = tri_arr[t_idx]
            if adj[i, j] and adj[j, k] and adj[k, i]:
                sums[t_idx] = deg[i] + deg[j] + deg[k]
            else:
                sums[t_idx] = 0
        return int(np.max(sums))

    def revert_swap(adj, sums, u, v, x, y):
        """Revert swap: add back (u,v), remove (x,y)."""
        adj[u, v] = 1
        adj[v, u] = 1
        adj[x, y] = 0
        adj[y, x] = 0
        deg = adj.sum(axis=1)
        affected = np.unique(np.concatenate([
            tri_incident_arr[u], tri_incident_arr[v],
            tri_incident_arr[x], tri_incident_arr[y]
        ]))
        for t_idx in affected:
            i, j, k = tri_arr[t_idx]
            if adj[i, j] and adj[j, k] and adj[k, i]:
                sums[t_idx] = deg[i] + deg[j] + deg[k]
            else:
                sums[t_idx] = 0
        return int(np.max(sums))

    def get_edges_non_edges(adj):
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        return edges, non_edges

    # Initialize: K_{8,8} + edge (0,1)
    current_adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            current_adj[u, v] = 1
            current_adj[v, u] = 1
    current_adj[0, 1] = 1
    current_adj[1, 0] = 1

    current_sums = compute_sums(current_adj)
    current_score = int(np.max(current_sums))
    best_adj = current_adj.copy()
    best_sums = current_sums.copy()
    best_score = current_score

    # Simulated Annealing
    T = 3.0
    cooling = 0.9995
    min_T = 0.3

    while time.time() - start_time < deadline - 20:
        edges, non_edges = get_edges_non_edges(current_adj)
        if not edges or not non_edges:
            break

        u, v = edges[np.random.randint(len(edges))]
        x, y = non_edges[np.random.randint(len(non_edges))]

        # Ensure disjoint
        if x in (u, v) or y in (u, v):
            continue

        old_score = current_score
        new_score = apply_swap(current_adj, current_sums, u, v, x, y)

        if new_score <= old_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
                best_sums = current_sums.copy()
        else:
            delta = new_score - old_score
            if np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                revert_swap(current_adj, current_sums, u, v, x, y)

        T *= cooling
        if T < min_T:
            T = min_T

    # Hill Climbing Polish
    improved = True
    while improved and time.time() - start_time < deadline - 5:
        improved = False
        edges, non_edges = get_edges_non_edges(best_adj)

        for u, v in edges:
            if time.time() - start_time > deadline - 5:
                break
            for x, y in non_edges:
                if x in (u, v) or y in (u, v):
                    continue

                # Try swap on best_adj
                best_adj[u, v] = 0
                best_adj[v, u] = 0
                best_adj[x, y] = 1
                best_adj[y, x] = 1

                # Recompute sums for best_adj
                deg = best_adj.sum(axis=1)
                valid = (best_adj[I, J] == 1) & (best_adj[J, K] == 1) & (best_adj[K, I] == 1)
                new_sums = np.zeros(n_tri, dtype=np.int32)
                new_sums[valid] = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
                new_score = int(np.max(new_sums))

                if new_score < best_score:
                    best_score = new_score
                    best_sums = new_sums
                    improved = True
                else:
                    # Revert
                    best_adj[u, v] = 1
                    best_adj[v, u] = 1
                    best_adj[x, y] = 0
                    best_adj[y, x] = 0

                if improved:
                    break
            if improved:
                break

    # Final verification and return
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])

    assert len(edges_list) == target_edges, f"Edge count {len(edges_list)} != {target_edges}"

    return {'n': n, 'edges': edges_list}
