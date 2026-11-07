# EVOLVE-BLOCK-START
import numpy as np
import time

def _optimize_tour_2opt(tour, distance_matrix):
    """
    Performs 2-opt optimization on a given tour.
    tour: list of node indices (customers only).
    """
    n = len(tour)
    if n <= 2:
        return tour
    
    current_tour = list(tour)
    
    def _tour_cost(t):
        total = 0.0
        prev = 0
        for node in t:
            total += distance_matrix[prev, node]
            prev = node
        total += distance_matrix[prev, 0]
        return total
    
    improved = True
    max_iterations = 20 # Reduced iterations for speed, as this is a heuristic generator
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                u = current_tour[i-1] if i > 0 else 0
                v = current_tour[i]
                x = current_tour[j]
                y = current_tour[j+1] if j < n - 1 else 0
                
                # Cost before: d(u,v) + d(x,y)
                # Cost after: d(u,x) + d(v,y)
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-10:
                    current_tour[i:j+1] = current_tour[i:j+1][::-1]
                    improved = True
                    
    return current_tour

def _construct_savings_routes(distance_matrix, demands, capacity):
    """
    Constructs routes using Clarke-Wright Savings heuristic.
    Returns a list of routes (lists of customer indices).
    """
    n = len(demands)
    # Initialize each customer as its own route
    routes = [[i] for i in range(1, n)]
    route_of_node = {i: i - 1 for i in range(1, n)} # node -> route index
    route_loads = [demands[i] for i in range(1, n)]
    
    # Calculate savings
    d_depot = distance_matrix[0, :]
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            savings[i, j] = d_depot[i] + d_depot[j] - distance_matrix[i, j]
            savings[j, i] = savings[i, j]
            
    # Sort pairs by savings in descending order
    pairs = []
    for i in range(1, n):
        for j in range(i + 1, n):
            if savings[i, j] > 1e-9:
                pairs.append((savings[i, j], i, j))
                
    pairs.sort(reverse=True)
    
    # Merge routes
    for s, i, j in pairs:
        ri = route_of_node[i]
        rj = route_of_node[j]
        
        if ri == rj:
            continue
            
        # Check if i and j are at the ends of their respective routes
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Find positions of i and j in their routes
        # i must be at start or end of route_i
        # j must be at start or end of route_j
        i_at_start = route_i[0] == i
        i_at_end = route_i[-1] == i
        j_at_start = route_j[0] == j
        j_at_end = route_j[-1] == j
        
        if not (i_at_start or i_at_end):
            continue
        if not (j_at_start or j_at_end):
            continue
            
        # Check capacity
        if route_loads[ri] + route_loads[rj] > capacity:
            continue
            
        # Merge routes
        # Determine order
        if i_at_start and j_at_start:
            new_route = route_j[::-1] + route_i
        elif i_at_start and j_at_end:
            new_route = route_j + route_i
        elif i_at_end and j_at_start:
            new_route = route_i + route_j
        elif i_at_end and j_at_end:
            new_route = route_i + route_j[::-1]
        else:
            continue # Should not happen due to checks above
            
        # Update data structures
        routes[ri] = new_route
        routes[rj] = [] # Mark as empty/merged
        route_loads[ri] += route_loads[rj]
        
        for node in new_route:
            route_of_node[node] = ri
            
    # Filter out empty routes
    valid_routes = [r for r in routes if r]
    return valid_routes

def _greedy_tour_cost(distance_matrix, heuristic_matrix, demands, capacity):
    """
    Simulates a greedy tour construction using the heuristic matrix.
    Returns the total distance.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_capacity = capacity
    total_distance = 0.0
    
    unvisited_count = n - 1
    while unvisited_count > 0:
        best_j = -1
        best_score = -1.0
        
        # Vectorized search for best feasible customer
        # Feasible: not visited, demand fits
        feasible_mask = (~visited) & (demands <= remaining_capacity)
        # Exclude depot just in case, though visited[0] is True
        feasible_mask[0] = False
        
        if not np.any(feasible_mask):
            # Return to depot
            total_distance += distance_matrix[current, 0]
            current = 0
            remaining_capacity = capacity
            continue
            
        # Get scores for feasible customers from current row
        scores = heuristic_matrix[current, :]
        # Mask out infeasible ones
        scores_feasible = np.where(feasible_mask, scores, -np.inf)
        
        best_j = np.argmax(scores_feasible)
        best_score = scores_feasible[best_j]
        
        if best_score <= 0:
             # Fallback to nearest neighbor if heuristic is weird
             dists = np.where(feasible_mask, distance_matrix[current, :], np.inf)
             best_j = np.argmin(dists)
             
        # Move to customer
        total_distance += distance_matrix[current, best_j]
        remaining_capacity -= demands[best_j]
        visited[best_j] = True
        unvisited_count -= 1
        current = best_j
        
    # Return to depot at the end
    total_distance += distance_matrix[current, 0]
    
    return total_distance

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix by selecting the best from a set of candidates
    based on a greedy proxy evaluation.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # Precompute common values
    eps = 1e-9
    safe_dist = distance_matrix + eps
    d_depot = distance_matrix[0, :]
    
    # Candidate 1: Inverse Distance
    h1 = 1.0 / safe_dist
    
    # Candidate 2: Savings-Enhanced Inverse Distance
    # S_ij = d_0i + d_0j - d_ij
    savings = d_depot[:, None] + d_depot[None, :] - distance_matrix
    savings = np.maximum(savings, 0.0)
    h2 = (1.0 + savings) / safe_dist
    
    # Candidate 3: Expert Boosted Savings-Enhanced
    # Generate expert solution using Savings + 2-Opt
    expert_routes = _construct_savings_routes(distance_matrix, demands, capacity)
    
    # Optimize each route with 2-Opt
    optimized_routes = []
    for route in expert_routes:
        # Only optimize if route has > 2 nodes and time allows
        if len(route) > 2 and time.time() < deadline - 10.0:
            opt_route = _optimize_tour_2opt(route, distance_matrix)
        else:
            opt_route = route
        optimized_routes.append(opt_route)
        
    # Construct Expert Edge Matrix
    expert_edges = np.zeros((n, n), dtype=bool)
    for route in optimized_routes:
        prev = 0
        for node in route:
            expert_edges[prev, node] = True
            expert_edges[node, prev] = True
            prev = node
        expert_edges[prev, 0] = True
        expert_edges[0, prev] = True
        
    # Boost factor: 50.0 seems a good balance
    boost_factor = 50.0
    h3 = h2 * (1.0 + boost_factor * expert_edges)
    
    # Ensure positivity
    h1 = np.maximum(h1, 1e-9)
    h2 = np.maximum(h2, 1e-9)
    h3 = np.maximum(h3, 1e-9)
    
    # Evaluate candidates using greedy proxy
    candidates = [h1, h2, h3]
    best_h = h1
    best_cost = float('inf')
    
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        # Check time
        if time.time() > deadline - 5.0:
            break
        cost = _greedy_tour_cost(distance_matrix, h, demands, capacity)
        if cost < best_cost:
            best_cost = cost
            best_h = h
            
    # Final safety check
    if not np.all(np.isfinite(best_h)):
        best_h = h1
        
    best_h = np.where(best_h > 0, best_h, 1e-9)
    
    return best_h
# EVOLVE-BLOCK-END
