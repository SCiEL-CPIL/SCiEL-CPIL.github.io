import numpy as np
import time

# Explicit deadline guard
_DEADLINE_FRACTION = 0.9
_WALL_LIMIT = 330.0
_start = time.time()

def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)

def _nearest_neighbor_tour(distance_matrix, start=0):
    """
    Construct a tour using the Nearest Neighbor heuristic starting from a specific node.
    """
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    current = start
    
    for _ in range(n - 1):
        # Find unvisited node with minimum distance
        # Mask visited nodes with inf
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return np.array(tour)

def _two_opt(tour, distance_matrix, max_iters=100):
    """
    Improve a tour using the 2-opt heuristic.
    """
    n = len(tour)
    tour = tour.copy()
    best_len = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
    
    for _ in range(max_iters):
        if _deadline_left() <= 0:
            break
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta
                # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                # New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                # Note: indices are mod n, but we handle the wrap-around carefully.
                # Standard 2-opt on a path/tour.
                
                a = tour[i]
                b = tour[i + 1]
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # If i=0, the edge is (tour[0], tour[1]) and (tour[n-1], tour[0])?
                # No, standard 2-opt reverses segment between i+1 and j.
                # Let's stick to the standard delta formula for a closed tour.
                # Edges removed: (a, b) and (c, d)
                # Edges added: (a, c) and (b, d)
                
                delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                
                if delta < -1e-9:
                    # Reverse segment between i+1 and j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    best_len += delta
                    improved = True
        if not improved:
            break
            
    return tour

def _good_directed_tour(distance_matrix):
    """
    Find a good directed tour by trying multiple starting nodes for NN + 2-opt.
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Try a few starting nodes (0, 1, 2, and maybe a random one if time permits)
    starts = [0, 1, 2]
    if n > 3:
        starts.append(n // 2)
        
    for start in starts:
        if _deadline_left() <= 0:
            break
        nn_tour = _nearest_neighbor_tour(distance_matrix, start)
        opt_tour = _two_opt(nn_tour, distance_matrix, max_iters=100)
        tour_len = sum(distance_matrix[opt_tour[i], opt_tour[(i + 1) % n]] for i in range(n))
        if tour_len < best_len:
            best_len = tour_len
            best_tour = opt_tour
            
    return best_tour

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a combination of inverse distance and a boost for edges
    present in a near-optimal tour found via NN + 2-opt.
    """
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero if any diagonal is 0 (though TSP usually has 0 on diagonal, 
    # but ants don't pick visited nodes, and i!=j. However, diagonal is 0 in distance matrix usually.
    # The task says "distance_matrix", usually symmetric with 0 diagonal.
    # We should ensure we don't divide by 0.
    d_safe = np.where(distance_matrix > 1e-9, distance_matrix, 1.0)
    base_heuristic = 1.0 / d_safe
    
    # Ensure diagonal is 0 or very small, as ants never pick themselves
    np.fill_diagonal(base_heuristic, 0.0)
    
    # Find a reference tour
    ref_tour = _good_directed_tour(distance_matrix)
    
    if ref_tour is None:
        ref_tour = _nearest_neighbor_tour(distance_matrix)
        
    # Create a mask for edges in the reference tour
    # Note: The tour is a sequence of nodes. Edges are (tour[i], tour[i+1])
    # Since the tour is closed, edges are (tour[i], tour[(i+1)%n])
    in_tour = np.zeros((n, n), dtype=float)
    for i in range(n):
        u = ref_tour[i]
        v = ref_tour[(i + 1) % n]
        in_tour[u, v] = 1.0
        in_tour[v, u] = 1.0 # Distance is symmetric, but heuristic is used in directed probability?
        # The task says "probability proportional to pheromone[i, j] * heuristic[i, j]"
        # and "pheromone[i, j] ... for each step i to j".
        # Usually ACO is directed, but TSP is symmetric.
        # The distance matrix is likely symmetric.
        # If the heuristic is symmetric, it's fine.
        
    # Boost edges in the reference tour
    # A multiplier of 5-10 is typically effective to guide without completely restricting
    boost_factor = 5.0
    heuristic = base_heuristic * (1.0 + boost_factor * in_tour)
    
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # Our heuristic is positive everywhere except diagonal (0).
    # We should set diagonal to 1e-9 or just leave it? 
    # "Heuristic entries that are not positive become 1e-9"
    # 0 is not positive. So diagonal should be 1e-9.
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
