import numpy as np
import time

def heuristics(distance_matrix):
    """
    Heuristic matrix for ant colony optimization of TSP.
    
    Uses a structural approach: compute a minimum spanning tree and use
    the tree edges as strong attractors, combined with a softened inverse
    distance base. The key insight is that TSP tours are closely related
    to MSTs, so boosting MST edges significantly guides ants toward
    feasible, low-cost tours.
    
    Additionally, we add a small boost to edges that are part of the
    2-core of the k-NN graph (edges that appear in both i's k-NN and
    j's k-NN), which captures mutual neighborhood structure.
    """
    start_time = time.time()
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Check for non-finite values
    if not np.all(np.isfinite(dm)):
        raise ValueError("Distance matrix contains non-finite values")
    
    # Deadline guard: return a simple heuristic if we're running low on time
    elapsed = time.time() - start_time
    if elapsed > 280:
        d_safe = dm + 1e-9
        heuristic = 1.0 / (d_safe ** 0.5)
        np.fill_diagonal(heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # Base heuristic: softened inverse distance
    alpha = 0.7
    d_safe = dm + 1e-9
    base_heuristic = 1.0 / (d_safe ** alpha)
    
    # Compute MST using Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0
    mst_edge = np.zeros((n, n))
    
    for _ in range(n):
        u = -1
        for i in range(n):
            if not in_mst[i] and (u == -1 or min_dist[i] < min_dist[u]):
                u = i
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and dm[u, v] < min_dist[v]:
                min_dist[v] = dm[u, v]
                mst_edge[u, v] = 1
                mst_edge[v, u] = 1
    
    # Strong boost for MST edges
    mst_boost = 1.0 + 2.0 * mst_edge
    
    # Mutual k-NN boost: edges where i is in j's k-NN and j is in i's k-NN
    k = 5
    mutual_knn = np.zeros((n, n))
    knn_sets = []
    for i in range(n):
        dists = dm[i].copy()
        dists[i] = np.inf
        knn_idx = set(np.argsort(dists)[:k])
        knn_sets.append(knn_idx)
    
    for i in range(n):
        for j in knn_sets[i]:
            if i in knn_sets[j]:
                mutual_knn[i, j] = 1
                mutual_knn[j, i] = 1
    
    knn_boost = 1.0 + 0.5 * mutual_knn
    
    # Combine
    heuristic = base_heuristic * mst_boost * knn_boost
    
    # Ensure diagonal is very small
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so the mean is around 1
    mean_val = heuristic[heuristic > 0].mean()
    if mean_val > 0:
        heuristic = heuristic / mean_val
    
    return heuristic
