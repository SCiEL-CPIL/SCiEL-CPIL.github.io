import numpy as np
import time

def _orient(a, b, c):
    """
    Orientation of triplet (a, b, c).
    1 if counter-clockwise, -1 if clockwise, 0 if collinear.
    a, b, c are tuples (x, y) or arrays [x, y].
    """
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(val) < 1e-10:
        return 0
    return 1 if val > 0 else -1

def _segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    Assumes no collinear overlapping segments (generic position).
    """
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def _nearest_neighbor_tour(dist, start=0):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        unvisited = ~visited
        unvisited[current] = True
        nearest = np.argmin(np.where(unvisited, dist[current], np.inf))
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    return tour

def _two_opt_tour(tour, dist, max_passes=50):
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

def _multi_start_local_search(dist, num_starts=5):
    n = dist.shape[0]
    best_tour = None
    best_len = float('inf')
    
    # Use a few different starting nodes
    starts = [0, n//2, n//3, 2*n//3, 1]
    for start in starts:
        nn_tour = _nearest_neighbor_tour(dist, start)
        opt_tour = _two_opt_tour(nn_tour, dist, max_passes=50)
        # Calculate length
        length = sum(dist[opt_tour[i]][opt_tour[(i+1)%n]] for i in range(n))
        if length < best_len:
            best_len = length
            best_tour = opt_tour
            
    return best_tour

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280
    
    # 1. Find a high-quality reference tour
    ref_tour = _multi_start_local_search(distance_matrix, num_starts=5)
    
    # 2. Extract edges of the reference tour
    # Store as list of tuples (u, v) with u < v to avoid duplicates? 
    # No, we need to check crossing with all segments.
    # A tour has N edges.
    ref_edges = []
    for i in range(n):
        u = ref_tour[i]
        v = ref_tour[(i + 1) % n]
        ref_edges.append((u, v))
        
    # Precompute coordinates for faster access?
    # distance_matrix is given, but we need coords for orientation.
    # Wait, the input is distance_matrix. We don't have coordinates!
    # The task says "50 uniform random points in the unit square".
    # But the function signature is heuristics(distance_matrix).
    # We do NOT have access to the coordinates!
    # We only have the distance matrix.
    
    # CRITICAL REALIZATION:
    # We cannot compute geometric crossings if we only have the distance matrix,
    # unless we reconstruct coordinates or use a property of distances.
    # However, the task says "Heuristic matrix for ant colony optimization...".
    # The evaluator likely provides just the distance matrix.
    # If we don't have coordinates, we cannot compute `_orient` and `_segments_intersect`.
    
    # Let's re-read the task carefully.
    # "Write heuristics(distance_matrix)..."
    # "3 fixed instances of 50 uniform random points in the unit square."
    # It does NOT say we receive the points. It says we receive the distance matrix.
    # Without coordinates, we cannot determine if two edges cross geometrically.
    # We can only use metric properties.
    
    # So the "planarity via crossing" strategy is INVALID because we lack coordinates.
    # We must stick to metric-based heuristics.
    
    # Alternative Planarity Proxy:
    # In a Euclidean TSP, the optimal tour is non-crossing.
    # Edges that are "long" relative to their neighborhood are likely to cross.
    # A common heuristic is $1/d$.
    # Another is using the k-NN graph.
    
    # Let's revert to a strong metric-based heuristic.
    # The current program uses:
    # base = 1/d^2
    # boost for ref tour edges
    # boost for k-NN edges
    # penalty for others
    
    # How to improve this?
    # 1. Better reference tour (Multi-start + 2-opt/3-opt).
    # 2. Smoother weighting.
    
    # Let's try:
    # 1. Multi-start NN + 2-opt to get `ref_tour`.
    # 2. Compute a "edge score" for all edges.
    #    - If edge is in `ref_tour`, score = 1.0
    #    - Else, score = 0.0
    # 3. Combine with distance:
    #    heuristic[i,j] = (1/d[i,j]) * (1 + alpha * score[i,j])
    #    This boosts good edges but keeps distance as primary driver.
    
    # To make it stronger, we can use multiple reference tours.
    # Generate K good tours.
    # Count frequency of each edge.
    # freq[i,j] = number of times edge (i,j) appears in the K tours.
    # heuristic[i,j] = (1/d[i,j]) * (1 + beta * freq[i,j])
    
    # This is robust and doesn't require coordinates.
    
    # Let's implement this "Frequency-based Heuristic".
    
    num_tours = 10
    tours = []
    for _ in range(num_tours):
        # Random start or different starts?
        # Let's use different starts to get diverse tours.
        start = np.random.randint(n) # But we need reproducibility? 
        # The task says "seeded per instance". The evaluator seeds numpy.
        # So np.random is safe.
        nn_tour = _nearest_neighbor_tour(distance_matrix, start)
        opt_tour = _two_opt_tour(nn_tour, distance_matrix, max_passes=50)
        tours.append(opt_tour)
        
    # Count edge frequencies
    freq = np.zeros((n, n), dtype=np.float64)
    for tour in tours:
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            freq[u, v] += 1
            freq[v, u] += 1
            
    # Normalize frequency
    # Max freq is num_tours.
    # We want to boost edges that appear in many tours.
    
    # Base heuristic: 1/d
    # Avoid division by zero
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Combine
    # heuristic = base * (1 + freq)
    # This means edges in all tours get 11x boost over distance-only.
    # Edges in no tours get 1x (just distance).
    
    # To make it more aggressive, we can use:
    # heuristic = base * (1 + freq ** 2)
    # or
    # heuristic = base * exp(freq)
    
    # Let's try: heuristic = base * (1 + 2.0 * freq)
    
    heuristic = base * (1.0 + 2.0 * freq)
    
    # Ensure finite and positive
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic

# EVOLVE-BLOCK-END
