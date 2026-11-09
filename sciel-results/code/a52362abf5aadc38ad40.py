"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _deadline_left(start_time, limit=325.0):
    """Check if we're within the time limit."""
    return (time.time() - start_time) < limit


def _nearest_neighbor_tour(dist):
    """Find a tour using nearest neighbor heuristic."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break
        dists = dist[current, unvisited]
        nearest = unvisited[np.argmin(dists)]
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    
    # Return to start
    tour.append(0)
    return tour


def _two_opt(tour, dist, max_passes=10):
    """Improve tour using 2-opt."""
    n = len(tour) - 1  # Exclude the final return to 0
    best_tour = tour.copy()
    best_length = sum(dist[tour[i], tour[i+1]] for i in range(n))
    
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate improvement
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[j+1]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    # Reverse segment i to j
                    best_tour[i:j+1] = best_tour[i:j+1][::-1]
                    best_length += delta
                    improved = True
        if not improved:
            break
    
    return best_tour, best_length


def _good_directed_tour(dist):
    """Find a good directed tour using NN + 2-opt."""
    nn_tour = _nearest_neighbor_tour(dist)
    optimized_tour, length = _two_opt(nn_tour, dist)
    return optimized_tour, length


def heuristics(distance_matrix):
    """Compute edge-attractiveness matrix for ACO TSP."""
    start_time = time.time()
    dist = np.array(distance_matrix, dtype=np.float64)
    n = dist.shape[0]
    
    # Ensure positive distances
    dist = np.maximum(dist, 1e-9)
    
    # Compute a good tour to use as reference
    good_tour, good_length = _good_directed_tour(dist)
    
    # Create edge mask for the good tour
    good_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(good_tour) - 1):
        good_edges[good_tour[i], good_tour[i+1]] = True
    
    # Base heuristic: inverse distance with power for stronger contrast
    # Use a power that creates meaningful differentiation
    base_heuristic = 1.0 / (dist + 1e-9) ** 1.5
    
    # Boost edges in the good tour significantly
    # This guides ACO toward promising regions
    boost_factor = 2.0
    heuristic = base_heuristic.copy()
    heuristic[good_edges] *= boost_factor
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize to prevent numerical issues
    heuristic = heuristic / heuristic.max()
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
