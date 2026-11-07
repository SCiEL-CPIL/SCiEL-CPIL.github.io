"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses a 2-opt improved nearest neighbor tour to create a strong structural bias.
    """
    n = distance_matrix.shape[0]
    
    # 1. Generate a good initial tour using Nearest Neighbor
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        current = tour[-1]
        # Find nearest unvisited node
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
    
    # 2. Improve the tour using 2-Opt
    def tour_length(tour, dist_matrix):
        length = 0.0
        for i in range(len(tour)):
            length += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return length

    def two_opt(tour, dist_matrix):
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    if i == 0 and j == len(tour) - 1:
                        continue
                    # Calculate delta length
                    # Edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                    # New edges: (i-1, j), (i, j+1)
                    a, b = tour[i-1], tour[i]
                    c, d = tour[j], tour[(j+1) % len(tour)]
                    
                    # Check if edges are the same (shouldn't happen in valid tour)
                    if (a == c and b == d) or (a == d and b == c):
                        continue

                    delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                    if delta < -1e-10:
                        # Reverse the segment between i and j
                        tour[i:j+1] = reversed(tour[i:j+1])
                        improved = True
        
        return tour

    tour = two_opt(tour, distance_matrix)
    
    # 3. Construct heuristic matrix
    # Base heuristic is inverse distance
    heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Create a mask for edges in the 2-opt tour
    in_tour = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        in_tour[u, v] = True
        in_tour[v, u] = True # Assuming symmetric TSP, but safe to mark both if input is symmetric
    
    # Boost edges in the tour significantly
    # This guides the ACO to strongly prefer the structure of the good tour
    boost = 100.0
    heuristic = heuristic * (1.0 + boost * in_tour)
    
    # Ensure non-finite values are handled (though 1/(d+eps) is safe)
    # The task says "Heuristic entries that are not positive become 1e-9"
    # Our heuristic is always positive.
    
    return heuristic
# EVOLVE-BLOCK-END
