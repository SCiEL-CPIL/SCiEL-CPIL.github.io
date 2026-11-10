import numpy as np
import time

def heuristics(distance_matrix):
    """
    Heuristic matrix for ant colony optimization of TSP.
    
    Combines a softened inverse distance base with multiple structural priors:
    1. Base: d^-0.7 (softened inverse to balance greediness and exploration)
    2. Heuristic Tour Boost: Strong boost to edges in a fast NN+2Opt tour
    3. MST Boost: Moderate boost to Minimum Spanning Tree edges
    4. Mutual k-NN Boost: Small boost to mutually close neighbors
    
    The heuristic tour provides a complete valid cycle example, guiding ants
    towards forming a single large loop. The MST provides global connectivity
    structure. The k-NN ensures local consistency.
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
        heuristic = 1.0 / (d_safe ** 0.7)
        np.fill_diagonal(heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # 1. Base heuristic: softened inverse distance
    alpha = 0.7
    d_safe = dm + 1e-9
    base_heuristic = 1.0 / (d_safe ** alpha)
    
    # 2. Compute MST using Prim's algorithm
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
    
    # 3. Compute Mutual k-NN boost
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
    
    # 4. Compute a fast heuristic tour (Nearest Neighbor + 2-Opt)
    # This provides a strong prior for edges that form a valid, short cycle.
    tour_edges = np.zeros((n, n))
    
    # Nearest Neighbor heuristic starting from node 0
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        next_node = -1
        min_d = np.inf
        for j in range(n):
            if not visited[j] and dm[current, j] < min_d:
                min_d = dm[current, j]
                next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # Close the tour
    tour.append(0)
    
    # Simple 2-Opt improvement to get a better tour structure
    # We limit iterations to ensure speed
    improved = True
    max_iter = 50
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Check if reversing segment i..j improves the tour
                # Edges involved: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1) % n]
                
                # Note: tour is 0-indexed in list, but tour[n] is 0
                # If j is n-1, j+1 is n which is index 0 in tour list? 
                # My tour list has length n+1, tour[n]=0.
                # So tour[j+1] is valid if j < n.
                
                old_dist = dm[a, b] + dm[c, d]
                new_dist = dm[a, c] + dm[b, d]
                
                if new_dist < old_dist - 1e-9:
                    # Reverse segment i to j
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    
    # Mark edges in the improved tour
    for i in range(n):
        u = tour[i]
        v = tour[(i+1) % n]
        tour_edges[u, v] = 1
        tour_edges[v, u] = 1
    
    # Combine boosts
    # Weights are tuned to prioritize the complete tour structure,
    # followed by global MST structure, then local k-NN.
    # Tour edges are the strongest signal for a valid cycle.
    tour_boost = 1.0 + 2.0 * tour_edges   # Factor 3 for tour edges
    mst_boost = 1.0 + 1.0 * mst_edge      # Factor 2 for MST edges
    knn_boost = 1.0 + 0.3 * mutual_knn    # Factor 1.3 for k-NN edges
    
    heuristic = base_heuristic * tour_boost * mst_boost * knn_boost
    
    # Ensure diagonal is very small
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so the mean is around 1
    mean_val = heuristic[heuristic > 0].mean()
    if mean_val > 0:
        heuristic = heuristic / mean_val
    
    return heuristic
