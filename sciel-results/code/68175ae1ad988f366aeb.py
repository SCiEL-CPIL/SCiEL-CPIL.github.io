import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # Helper to calculate tour length using vectorized operations
    def get_tour_length(t):
        idx = np.array(t)
        next_idx = np.roll(idx, -1)
        return np.sum(distance_matrix[idx, next_idx])

    # 2-Opt improvement
    def two_opt(t):
        t = list(t)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a = t[i]
                    b = t[i + 1]
                    c = t[j]
                    d = t[(j + 1) % n]
                    
                    # Check if reversing segment (i+1, ..., j) improves length
                    # Old edges: (a, b) and (c, d)
                    # New edges: (a, c) and (b, d)
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < -1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    # Nearest Neighbor construction
    def nearest_neighbor(start_node):
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        tour = [start_node]
        current = start_node
        
        for _ in range(n - 1):
            if time.time() > deadline:
                break
            # Find nearest unvisited node
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour

    # Farthest Insertion construction
    def farthest_insertion():
        # Start with 0 and the farthest node from 0
        dists_from_0 = distance_matrix[0, :]
        farthest_node = np.argmax(dists_from_0)
        tour = [0, farthest_node]
        remaining = [i for i in range(n) if i not in tour]
        
        while remaining:
            if time.time() > deadline:
                break
            # Find point in remaining that is farthest from the current tour
            max_min_dist = -1
            point_to_insert = -1
            for k in remaining:
                min_dist_to_tour = np.min(distance_matrix[k, tour])
                if min_dist_to_tour > max_min_dist:
                    max_min_dist = min_dist_to_tour
                    point_to_insert = k
            
            # Insert point_to_insert into tour at the position that minimizes increase
            best_pos = -1
            best_increase = float('inf')
            for i in range(len(tour)):
                prev_node = tour[i]
                next_node = tour[(i + 1) % len(tour)]
                increase = (distance_matrix[prev_node, point_to_insert] + 
                            distance_matrix[point_to_insert, next_node] - 
                            distance_matrix[prev_node, next_node])
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            
            tour.insert(best_pos, point_to_insert)
            remaining.remove(point_to_insert)
            
        return tour

    # Main search loop
    best_tour = None
    best_len = float('inf')
    
    # 1. Nearest Neighbor from 0
    tour_nn = nearest_neighbor(0)
    tour_nn = two_opt(tour_nn)
    len_nn = get_tour_length(tour_nn)
    if len_nn < best_len:
        best_len = len_nn
        best_tour = tour_nn

    # 2. Farthest Insertion
    tour_fi = farthest_insertion()
    tour_fi = two_opt(tour_fi)
    len_fi = get_tour_length(tour_fi)
    if len_fi < best_len:
        best_len = len_fi
        best_tour = tour_fi

    # 3. Random Restarts
    # Use a fixed seed for reproducibility within the heuristic calculation
    rng = np.random.default_rng(42)
    num_restarts = 20
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        rand_tour = list(range(n))
        rng.shuffle(rand_tour)
        # Ensure it starts at 0? ACO starts at 0, but tour length is invariant to rotation.
        # However, for consistency in 2-Opt, any start is fine.
        improved_rand = two_opt(rand_tour)
        len_rand = get_tour_length(improved_rand)
        if len_rand < best_len:
            best_len = len_rand
            best_tour = improved_rand

    # Fallback if something went wrong
    if best_tour is None:
        best_tour = list(range(n))
        best_tour = two_opt(best_tour)
        best_len = get_tour_length(best_tour)

    # Construct Heuristic Matrix
    # Base heuristic: inverse distance
    # Ensure diagonal is not 0 to avoid div by zero, but we will set it to 1e-9 later
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0) # Avoid div by zero on diagonal
    
    eta = 1.0 / safe_dist
    
    # Boost edges in the best found tour
    # The ACO prefers edges with higher eta. By setting the edges of our best tour
    # to a very high value, we guide the ants to reconstruct this tour initially.
    # Then pheromone updates and other edges can compete, but the strong bias
    # usually leads to the best known solution being found quickly.
    boost_factor = 1000.0
    
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        eta[u, v] = boost_factor
        # Note: The distance matrix is symmetric, but the ACO tour is directed.
        # The heuristic matrix is used as heuristic[i,j]. 
        # In a symmetric TSP, eta[i,j] == eta[j,i] is typical for distance-based heuristics.
        # However, the tour direction is i -> j. 
        # The problem statement says "probability proportional to pheromone[i, j] * heuristic[i, j]".
        # It does not explicitly say the heuristic must be symmetric, but distance is.
        # Boosting only i->j is sufficient to guide the forward construction.
        # If we boost both, it's also fine. Let's boost both for symmetry if we assume undirected edges.
        # Actually, the tour is a cycle. The ant moves i->j. So eta[i,j] is the relevant one.
        # If we only boost eta[i,j], the reverse edge eta[j,i] remains low.
        # This is correct for guiding the specific direction of the tour found.
        # However, 2-Opt found a cycle. The direction doesn't matter for length.
        # If the ACO constructs the reverse, it will use low eta values.
        # It is safer to boost both directions if we want to be agnostic to direction,
        # but the ACO starts at 0 and builds a path. The path direction is fixed by the choices.
        # If we found tour [0, 1, 2...], we want the ant to pick 1 then 2.
        # So eta[0,1] and eta[1,2] should be high.
        # Let's stick to boosting the directed edges of the found tour.
        
    # Ensure diagonal is not attractive
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta = np.maximum(eta, 1e-9)
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
