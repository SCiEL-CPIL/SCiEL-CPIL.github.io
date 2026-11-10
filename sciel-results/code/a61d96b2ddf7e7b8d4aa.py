"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dist):
    """Construct a nearest-neighbor tour starting from node 0."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        unvisited = ~visited
        unvisited[current] = True
        nearest = np.argmin(np.where(unvisited, dist[current], np.inf))
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    return tour


def _two_opt_tour(tour, dist, max_passes=10):
    """Improve tour using 2-opt local search."""
    n = len(tour)
    if n < 4:
        return tour
    
    best_tour = list(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                old_len = dist[a][b] + dist[c][d]
                new_len = dist[a][c] + dist[b][d]
                if new_len < old_len - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    improved = True
    return best_tour


def _or_opt_tour(tour, dist, max_passes=5):
    """Improve tour using Or-opt local search (move single nodes or pairs)."""
    n = len(tour)
    if n < 5:
        return tour
    
    best_tour = list(tour)
    
    for _ in range(max_passes):
        improved = False
        # Try moving single nodes
        for i in range(n):
            if i == n - 1:
                continue
            node = best_tour[i]
            prev_node = best_tour[i - 1]
            next_node = best_tour[i + 1]
            
            # Remove node from position i
            # Edges affected: (prev_node, node), (node, next_node)
            # New edges: (prev_node, next_node)
            
            for j in range(n - 1):
                if j == i or j == i - 1:
                    continue
                # Insert node between best_tour[j] and best_tour[j+1]
                # But need to be careful with indices after removal
                # Let's use a different approach: try all insertion positions
                
                # Actually, let's just try inserting at each position
                # This is O(n^2) per pass which is fine for n=50
                
        # Simpler approach: for each node, try moving it to each other position
        for i in range(n):
            if i == n - 1:
                continue
            node = best_tour[i]
            prev_idx = i - 1
            next_idx = i + 1
            
            # Calculate cost of removing node from position i
            # Remove edges (prev, node) and (node, next), add edge (prev, next)
            cost_remove = dist[best_tour[prev_idx]][node] + dist[node][best_tour[next_idx]] - dist[best_tour[prev_idx]][best_tour[next_idx]]
            
            # Try inserting at each position j
            for j in range(n - 1):
                if j == i or j == i - 1:
                    continue
                # Insert between best_tour[j] and best_tour[j+1]
                # But if j > i, indices shift after removal
                # Let's handle this carefully
                
                # Create a copy to test
                test_tour = list(best_tour)
                # Remove node at position i
                test_tour.pop(i)
                
                # Adjust j if necessary
                if j > i:
                    j -= 1
                
                # Calculate cost of inserting at position j in test_tour
                if j == len(test_tour) - 1:
                    # Insert at end (before wrapping to start)
                    insert_prev = test_tour[j]
                    insert_next = test_tour[0]
                else:
                    insert_prev = test_tour[j]
                    insert_next = test_tour[j + 1]
                
                cost_insert = dist[insert_prev][node] + dist[node][insert_next] - dist[insert_prev][insert_next]
                
                total_change = cost_remove + cost_insert
                if total_change < -1e-12:
                    # Apply the move
                    best_tour.pop(i)
                    if j > i:
                        j -= 1
                    best_tour.insert(j + 1, node)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_tour


def _three_opt_tour(tour, dist, max_passes=3):
    """Improve tour using 3-opt local search."""
    n = len(tour)
    if n < 5:
        return tour
    
    best_tour = list(tour)
    best_len = sum(dist[best_tour[i]][best_tour[(i+1) % n]] for i in range(n))
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if i == 0 and k == n - 1:
                        continue
                    a, b = best_tour[i], best_tour[i + 1]
                    c, d = best_tour[j], best_tour[j + 1]
                    e, f = best_tour[k], best_tour[(k + 1) % n]
                    # Try reversing segment [i+1, j]
                    # Original: a-b ... c-d ... e-f
                    # New: a-c ... b-d ... e-f (reverse i+1 to j)
                    # Actually 3-opt tries different combinations
                    # Let's do a simpler 3-opt: try all ways to reconnect 3 edges
                    
                    # Edge 1: (a, b), Edge 2: (c, d), Edge 3: (e, f)
                    # Option 1: a-c, b-d, e-f (reverse segment i+1..j)
                    # Option 2: a-d, b-c, e-f
                    # Option 3: a-c, b-e, d-f (reverse segment j+1..k)
                    # etc.
                    
                    # Simplified: just try reversing segment [i+1, j]
                    segment = best_tour[i+1:j+1]
                    reversed_segment = segment[::-1]
                    
                    # Calculate new length
                    new_len = best_len - dist[a][b] - dist[c][d] + dist[a][reversed_segment[0]] + dist[reversed_segment[-1]][d]
                    
                    if new_len < best_len - 1e-12:
                        # Apply the change
                        best_tour[i+1:j+1] = reversed_segment
                        best_len = new_len
                        improved = True
        if not improved:
            break
    return best_tour


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Combines:
    1. Inverse distance (power 2) as base
    2. Reference tour boost (NN + 2-opt + 3-opt) - strong attraction for good edges
    3. k-NN graph boost
    4. Local scale penalty for long edges
    """
    n = distance_matrix.shape[0]
    deadline = time.time() + 280
    
    # Build reference tour: NN + 2-opt + Or-opt + 3-opt
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    ref_tour = _two_opt_tour(nn_tour, distance_matrix, max_passes=10)
    ref_tour = _or_opt_tour(ref_tour, distance_matrix, max_passes=5)
    ref_tour = _three_opt_tour(ref_tour, distance_matrix, max_passes=3)
    # One more 2-opt pass to polish
    ref_tour = _two_opt_tour(ref_tour, distance_matrix, max_passes=5)
    
    # Create reference tour edge mask
    ref_tour_edges = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = ref_tour[i]
        b = ref_tour[(i + 1) % n]
        ref_tour_edges[a, b] = 1.0
        ref_tour_edges[b, a] = 1.0
    
    # Base: inverse distance with power 2.5 (slightly sharper than power 2)
    base = 1.0 / (distance_matrix + 1e-9) ** 2.5
    
    # Component 1: Strong reference tour boost
    # Edges in the reference tour get a large multiplier
    ref_multiplier = 1.0 + 4.0 * ref_tour_edges
    
    # Component 2: k-NN graph boost (moderate)
    k = 6
    knn_boost = np.zeros_like(distance_matrix)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for rank, j in enumerate(nn_indices):
            weight = 1.0 - (rank / k)
            knn_boost[i, j] = weight
            knn_boost[j, i] = max(knn_boost[j, i], weight)
    
    knn_multiplier = 1.0 + 0.6 * knn_boost
    
    # Component 3: Local scale penalty
    # Precompute local scales
    local_scales = np.zeros(n)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        k_local = min(10, n - 1)
        sorted_dists = np.sort(dists)[:k_local]
        local_scales[i] = np.median(sorted_dists)
    
    # Long-edge penalty
    crossing_penalty = np.ones_like(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = distance_matrix[i, j]
            local_scale = min(local_scales[i], local_scales[j])
            if local_scale < 1e-9:
                local_scale = 1e-9
            ratio = dist_ij / local_scale
            if ratio > 1.3:
                penalty = np.exp(-0.6 * (ratio - 1.3))
                crossing_penalty[i, j] = penalty
                crossing_penalty[j, i] = penalty
    
    # Combine all components
    heuristic = base * ref_multiplier * knn_multiplier * crossing_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
