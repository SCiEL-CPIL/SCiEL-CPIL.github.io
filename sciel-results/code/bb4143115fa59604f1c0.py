import numpy as np

def _nearest_neighbor_tour(dist):
    n = dist.shape[0]
    unvisited = set(range(n))
    unvisited.remove(0)
    tour = [0]
    current = 0
    while unvisited:
        # Find nearest
        neighbors = list(unvisited)
        # Vectorized find min
        dists = dist[current, neighbors]
        next_node = neighbors[np.argmin(dists)]
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour

def _two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Avoid immediate neighbors in the tour cycle logic if needed, 
                # but standard 2-opt checks all i, j with i < j
                if j == i + 1:
                    continue
                # Calculate cost difference
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[(j+1)%n]
                
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse segment tour[i..j]
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    return tour

def heuristics(distance_matrix):
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]
    
    # Base heuristic: inverse distance
    # Add small epsilon to avoid division by zero for self-loops if any, 
    # though TSP usually has 0 on diagonal.
    epsilon = 1e-9
    base_heur = 1.0 / (dist + epsilon)
    
    # Zero out diagonal (self-loops) to prevent ants from staying put
    np.fill_diagonal(base_heur, 0.0)
    
    # Generate reference tour
    nn_tour = _nearest_neighbor_tour(dist)
    opt_tour = _two_opt(nn_tour, dist)
    
    # Boost heuristic for edges in the optimized tour
    # We want to make these edges more attractive
    boost_factor = 3.0
    
    for i in range(n):
        u = opt_tour[i]
        v = opt_tour[(i + 1) % n]
        base_heur[u, v] *= boost_factor
        base_heur[v, u] *= boost_factor # Heuristic is symmetric in TSP usually, 
                                        # but ACO uses directed pheromone. 
                                        # The heuristic matrix is typically symmetric.
    
    # Ensure all values are positive and finite
    # The problem statement says: "Heuristic entries that are not positive become 1e-9"
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9
    
    return base_heur
