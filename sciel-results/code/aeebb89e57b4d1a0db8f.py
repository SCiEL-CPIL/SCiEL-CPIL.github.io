import numpy as np
import time

def _two_opt(tour, d):
    """
    Optimizes a tour using 2-opt.
    tour: list of node indices (0 to n-1, closed loop implied by tour[0] at end or just sequence)
    d: distance matrix
    Returns optimized tour and length.
    """
    n = len(tour)
    if n <= 2:
        # Calculate length for small tours
        length = 0.0
        for i in range(n):
            length += d[tour[i], tour[(i+1)%n]]
        return tour, length
    
    # Convert to list for easier manipulation if needed, but we can work in place
    # Standard 2-opt: reverse segment tour[i+1..j]
    # We work with 0-indexed nodes in the tour list.
    # Edge (i, i+1) and (j, j+1) are replaced by (i, j) and (i+1, j+1)
    # Note: indices in tour list.
    
    current_tour = list(tour)
    
    def tour_length(t):
        return sum(d[t[i], t[(i+1)%n]] for i in range(n))
    
    current_len = tour_length(current_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Skip trivial cases
                if i == 0 and j == n - 1:
                    continue
                # Nodes involved:
                # u = current_tour[i-1] (if i>0 else current_tour[n-1]) -- wait, standard 2-opt on path
                # Let's use the standard definition:
                # Remove edges (a, b) and (c, d) where a=current_tour[i-1], b=current_tour[i], c=current_tour[j], d=current_tour[j+1]
                # Add edges (a, c) and (b, d)
                # This reverses the segment current_tour[i..j]
                
                if i == 0:
                    a = current_tour[n-1]
                    b = current_tour[0]
                else:
                    a = current_tour[i-1]
                    b = current_tour[i]
                    
                if j == n-1:
                    c = current_tour[n-1]
                    d_node = current_tour[0]
                else:
                    c = current_tour[j]
                    d_node = current_tour[j+1]
                
                # Old cost: d[a,b] + d[c,d_node]
                # New cost: d[a,c] + d[b,d_node]
                old_cost = d[a, b] + d[c, d_node]
                new_cost = d[a, c] + d[b, d_node]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse segment i to j
                    current_tour[i:j+1] = current_tour[i:j+1][::-1]
                    current_len += new_cost - old_cost
                    improved = True
                    # Optionally break inner loop if we want first-improvement, 
                    # but best-improvement or continuing scan is standard.
                    # For speed, we can break if we just did a move, but continuing 
                    # to find more improvements in same pass is okay for n=50.
                    # Actually, restarting the inner loop or just continuing is fine.
                    # Let's just mark improved and continue.
    
    return current_tour, current_len

def _generate_random_tour(n, rng):
    tour = list(range(n))
    rng.shuffle(tour)
    return tour

def _nearest_neighbor_tour(n, d, start=0):
    visited = [False] * n
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                dist = d[current, j]
                if dist < min_dist:
                    min_dist = dist
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _farthest_neighbor_tour(n, d, start=0):
    visited = [False] * n
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        max_dist = -1
        next_node = -1
        for j in range(n):
            if not visited[j]:
                dist = d[current, j]
                if dist > max_dist:
                    max_dist = dist
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 250.0 # Safety margin
    
    d = distance_matrix
    n = d.shape[0]
    
    # Initialize heuristic matrix
    # We will accumulate a score for each edge (i, j)
    # score[i, j] += weight if edge (i, j) is in the optimized tour
    # Then normalize or transform to attractiveness.
    
    edge_count = np.zeros((n, n))
    
    # Generate multiple diverse tours
    rng = np.random.default_rng(42)
    
    # 1. Random tours
    for _ in range(5):
        tour = _generate_random_tour(n, rng)
        opt_tour, opt_len = _two_opt(tour, d)
        # Add to edge count
        for i in range(n):
            u = opt_tour[i]
            v = opt_tour[(i+1)%n]
            edge_count[u, v] += 1.0
            # Since TSP is symmetric in distance but ACO uses directed pheromone?
            # "in travel direction only" for pheromone update.
            # Heuristic is usually symmetric if distance is. 
            # The prompt says "edge-attractiveness matrix". 
            # If d is symmetric, h should probably be symmetric or at least consistent.
            # ACO uses h[i,j] for transition i->j.
            # If we count undirected edges, we should probably set both h[i,j] and h[j,i].
            # However, the 2-opt tour has a direction. 
            # Let's count directed edges as they appear in the tour.
            # Actually, for a symmetric TSP, the optimal tour is a cycle. 
            # Direction doesn't matter for length.
            # But ACO builds a directed path.
            # If we bias h[i,j] high for edges in the tour, we help ants traverse them.
            # Since the tour is a cycle, every edge is traversed in one direction in the tour list.
            # But the ant could go the other way? No, the ant builds a specific path.
            # If we only boost the direction in the tour, we might hinder the reverse direction.
            # However, 2-opt on a symmetric matrix produces a tour. The reverse tour is equally good.
            # It's safer to boost both directions for symmetric edges, or just follow the tour direction.
            # Let's boost both to be safe, as "attractiveness" of an edge usually implies bidirectional utility in heuristic construction unless specified otherwise.
            # Wait, the prompt says "edge-attractiveness matrix".
            # If I set h[i,j] high, it helps i->j.
            # Let's set both h[i,j] and h[j,i] to be robust.
            edge_count[v, u] += 1.0

    # 2. Nearest Neighbor tours from different starts
    for start in range(0, n, 10): # 5 starts
        tour = _nearest_neighbor_tour(n, d, start)
        opt_tour, opt_len = _two_opt(tour, d)
        for i in range(n):
            u = opt_tour[i]
            v = opt_tour[(i+1)%n]
            edge_count[u, v] += 1.0
            edge_count[v, u] += 1.0

    # 3. Farthest Neighbor tours from different starts
    for start in range(0, n, 10): # 5 starts
        tour = _farthest_neighbor_tour(n, d, start)
        opt_tour, opt_len = _two_opt(tour, d)
        for i in range(n):
            u = opt_tour[i]
            v = opt_tour[(i+1)%n]
            edge_count[u, v] += 1.0
            edge_count[v, u] += 1.0

    # 4. Greedy construction (closest to previous)
    # This is basically NN, but let's add a few more random restarts for diversity
    
    # Check time
    if time.time() > deadline - 50:
        # If we are running out of time, stop adding tours
        pass
    else:
        # Add 5 more random tours
        for _ in range(5):
            tour = _generate_random_tour(n, rng)
            opt_tour, opt_len = _two_opt(tour, d)
            for i in range(n):
                u = opt_tour[i]
                v = opt_tour[(i+1)%n]
                edge_count[u, v] += 1.0
                edge_count[v, u] += 1.0

    # Convert edge_count to heuristic
    # We want larger values to be more attractive.
    # Edges that appear in many good tours should be attractive.
    # Also, shorter edges are generally better.
    # Let's combine: Attractiveness = (1 + edge_count) / distance
    # Or: Attractiveness = edge_count * (1/distance)
    # If edge_count is 0, attractiveness should be low but positive (fallback to distance).
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (d + 1e-9)
    
    # Boost based on consensus
    # Normalize edge_count? 
    # Max count is roughly 15-20.
    # Let's use: h = base_heuristic * (1 + edge_count)
    # This ensures that edges in good tours are boosted, but distance still matters.
    
    # Alternatively, h = base_heuristic + edge_count * base_heuristic
    # Let's try: h = base_heuristic * (1 + edge_count / max_count)
    # This normalizes the boost.
    
    max_count = np.max(edge_count)
    if max_count == 0:
        max_count = 1.0
    
    # Boost factor: 1 for unused, up to (1 + 1) = 2 for most used? 
    # Let's make it stronger: 1 + 5 * (edge_count / max_count)
    # So unused edges have factor 1, best edges have factor 6.
    
    boost_factor = 1.0 + 5.0 * (edge_count / max_count)
    
    h = base_heuristic * boost_factor
    
    # Ensure diagonal is 0 or very small? 
    # The prompt says "Heuristic entries that are not positive become 1e-9".
    # Diagonals are distance 0, so 1/0 is inf. 
    # We must handle diagonals.
    # In ACO, you don't go to the same node. 
    # But the matrix must be valid.
    # Set diagonal to 1e-9 or 0? 
    # "Heuristic entries that are not positive become 1e-9".
    # If we set diagonal to 0, it becomes 1e-9.
    # If we set diagonal to inf, it stays inf.
    # Inf is not finite. "non-finite values ... make the whole result invalid".
    # So we MUST avoid inf.
    
    # d[i,i] = 0. 1/0 = inf.
    # So we must mask diagonals.
    
    np.fill_diagonal(h, 1e-9)
    
    # Ensure all values are finite and positive
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    return h
