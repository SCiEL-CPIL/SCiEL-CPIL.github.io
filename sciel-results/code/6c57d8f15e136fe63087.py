"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(n):
    """Generate a tour using the nearest neighbor heuristic."""
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    curr = 0
    for _ in range(n - 1):
        # Find nearest unvisited node
        dists = np.where(visited, np.inf, np.linalg.norm(POINTS - POINTS[curr], axis=1))
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        curr = next_node
    return tour

def _two_opt(tour):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing segment i+1 to j
                # Current: ... A-B ... C-D ...
                # New: ... A-C ... B-D ...
                # Points: tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                
                # Distances in Euclidean space
                # d(a,b) + d(c,d) vs d(a,c) + d(b,d)
                # Since we don't have a distance matrix passed to this helper easily 
                # without global state or passing it, we use POINTS global.
                # However, the task provides distance_matrix to heuristics().
                # Let's refactor to use distance_matrix passed down or global.
                # For efficiency, we'll use the global POINTS calculated in heuristics.
                
                da_b = np.linalg.norm(POINTS[a] - POINTS[b])
                dc_d = np.linalg.norm(POINTS[c] - POINTS[d])
                da_c = np.linalg.norm(POINTS[a] - POINTS[c])
                db_d = np.linalg.norm(POINTS[b] - POINTS[d])
                
                if da_c + db_d < da_b + dc_d - 1e-10:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    return tour

def _compute_consensus_heuristic(distance_matrix, max_starts=100):
    """
    Compute a heuristic matrix based on the frequency and quality of edges
    appearing in multiple 2-opt optimized tours from different starts.
    """
    global POINTS
    n = distance_matrix.shape[0]
    
    # Reconstruct points from distance matrix? 
    # No, we can't easily invert distance matrix to points.
    # But the task says "50 uniform random points in the unit square".
    # The input is distance_matrix.
    # Wait, _two_opt uses POINTS. I need to either pass distance_matrix to _two_opt
    # or reconstruct points. Reconstructing is hard.
    # Let's modify _two_opt to take distance_matrix.
    
    # Actually, let's just use distance_matrix for distance lookups in _two_opt.
    
    # Re-define helper to use distance_matrix
    def _two_opt_dm(tour, dm):
        n = len(tour)
        tour = list(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    
                    da_b = dm[a, b]
                    dc_d = dm[c, d]
                    da_c = dm[a, c]
                    db_d = dm[b, d]
                    
                    if da_c + db_d < da_b + dc_d - 1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
        return tour

    # Nearest neighbor using distance matrix
    def _nn_tour(dm):
        n = dm.shape[0]
        visited = np.zeros(n, dtype=bool)
        tour = [0]
        visited[0] = True
        curr = 0
        for _ in range(n - 1):
            dists = np.where(visited, np.inf, dm[curr])
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            curr = next_node
        return tour

    # Edge score accumulator
    edge_scores = np.zeros((n, n), dtype=np.float64)
    
    # Time budget for heuristic calculation: 10 seconds max
    start_time = time.time()
    time_limit = 10.0
    
    num_starts = 0
    for _ in range(max_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random starting node for diversity? 
        # Or just use nearest neighbor from node 0, but perturb?
        # Nearest neighbor from 0 is deterministic. 
        # To get diversity, we can start from random nodes or use random insertion.
        # Let's start from random nodes.
        start_node = np.random.randint(n)
        
        # Nearest neighbor from start_node
        visited = np.zeros(n, dtype=bool)
        tour = [start_node]
        visited[start_node] = True
        curr = start_node
        for _ in range(n - 1):
            dists = np.where(visited, np.inf, distance_matrix[curr])
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            curr = next_node
            
        # Optimize with 2-opt
        optimized_tour = _two_opt_dm(tour, distance_matrix)
        
        # Calculate tour length
        tour_len = 0.0
        for i in range(n):
            u = optimized_tour[i]
            v = optimized_tour[(i + 1) % n]
            tour_len += distance_matrix[u, v]
            
        # Accumulate score: 1/length for each edge in the tour
        score = 1.0 / tour_len
        for i in range(n):
            u = optimized_tour[i]
            v = optimized_tour[(i + 1) % n]
            edge_scores[u, v] += score
            edge_scores[v, u] += score
            
        num_starts += 1
        
    # Normalize or transform
    # If no tours were generated (shouldn't happen), fallback
    if num_starts == 0:
        return 1.0 / (distance_matrix + 1e-9)
        
    # The heuristic should be "larger is more attractive".
    # edge_scores are sums of 1/length. Higher means more frequent in good tours.
    # We can use the scores directly, or apply a power law to sharpen.
    # Let's try a moderate power to emphasize strong edges.
    heuristic = edge_scores ** 1.5
    
    # Ensure no zeros become 0 (task says non-positive become 1e-9, but we can avoid 0s)
    heuristic[heuristic == 0] = 1e-9
    
    return heuristic

def heuristics(distance_matrix):
    # Set global POINTS if needed, but we refactored to use DM directly.
    # Just in case, clear any global state dependency.
    return _compute_consensus_heuristic(distance_matrix)
# EVOLVE-BLOCK-END
