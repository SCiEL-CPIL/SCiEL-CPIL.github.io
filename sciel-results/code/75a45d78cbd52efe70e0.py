import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    n = distance_matrix.shape[0]
    customers = np.arange(1, n)
    
    # --- 1. Lagrangian Relaxation: Solve relaxed TSP to get edge preferences ---
    # The relaxed problem ignores capacity constraints but penalizes load violations
    # via dual variables. We approximate this by finding a good TSP tour and using
    # its edges as high-attractiveness candidates.
    
    def _nearest_neighbor_tour(dm, seed=0):
        """Find a TSP tour using nearest neighbor with random restart."""
        rng = np.random.RandomState(seed)
        unvisited = list(range(1, n))
        rng.shuffle(unvisited)
        
        # Start from depot
        tour = [0]
        current = 0
        remaining = unvisited.copy()
        
        while remaining:
            # Find nearest unvisited node
            best_node = min(remaining, key=lambda x: dm[current, x])
            tour.append(best_node)
            current = best_node
            remaining.remove(best_node)
        
        tour.append(0)  # Return to depot
        return tour
    
    def _two_opt_tour(tour, dm):
        """Apply 2-Opt to a TSP tour."""
        improved = True
        while improved:
            improved = False
            n_tour = len(tour)
            for i in range(1, n_tour - 2):
                for j in range(i + 1, n_tour - 1):
                    u, v = tour[i-1], tour[i]
                    w, x = tour[j], tour[j+1]
                    old_dist = dm[u, v] + dm[w, x]
                    new_dist = dm[u, w] + dm[v, x]
                    if new_dist < old_dist - 1e-10:
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        return tour
    
    # Find multiple TSP tours and aggregate edge frequencies
    n_tsp_starts = 20
    tsp_edge_matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n_tsp_starts):
        if time.time() > deadline - 150:
            break
        tour = _nearest_neighbor_tour(distance_matrix, seed=i)
        tour = _two_opt_tour(tour, distance_matrix)
        
        # Extract edges
        for k in range(len(tour) - 1):
            u, v = tour[k], tour[k+1]
            tsp_edge_matrix[u, v] += 1.0
            tsp_edge_matrix[v, u] += 1.0  # Symmetric
    
    # --- 2. Capacity-Aware Refinement: Use savings heuristic to identify feasible edges ---
    def _savings_routes(dm, demands, cap, seed):
        """Construct routes using savings heuristic with capacity constraints."""
        rng = np.random.RandomState(seed)
        unvisited = list(range(1, n))
        rng.shuffle(unvisited)
        routes = []
        
        while unvisited:
            route = [0]
            current = 0
            remaining_cap = cap
            start_idx = rng.randint(0, len(unvisited))
            first_node = unvisited.pop(start_idx)
            if demands[first_node] > remaining_cap:
                continue
            route.append(first_node)
            current = first_node
            remaining_cap -= demands[first_node]
            
            while unvisited:
                best_node = -1
                best_dist = np.inf
                for node in unvisited:
                    if demands[node] <= remaining_cap:
                        d = dm[current, node]
                        if d < best_dist:
                            best_dist = d
                            best_node = node
                if best_node == -1:
                    break
                route.append(best_node)
                remaining_cap -= demands[best_node]
                current = best_node
                unvisited.remove(best_node)
            
            route.append(0)
            if len(route) > 2:
                routes.append(route)
        
        return routes
    
    def _two_opt_route(route, dm, demands, cap):
        """Apply 2-Opt to a single route with capacity check."""
        improved = True
        while improved:
            improved = False
            n_route = len(route)
            for i in range(1, n_route - 2):
                for j in range(i + 1, n_route - 1):
                    u, v = route[i-1], route[i]
                    w, x = route[j], route[j+1]
                    old_dist = dm[u, v] + dm[w, x]
                    new_dist = dm[u, w] + dm[v, x]
                    if new_dist < old_dist - 1e-10:
                        # Check capacity after reversal
                        test_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        load = 0
                        valid = True
                        for node in test_route[1:-1]:
                            load += demands[node]
                            if load > cap:
                                valid = False
                                break
                        if valid:
                            route[i:j+1] = route[i:j+1][::-1]
                            improved = True
        return route
    
    # Generate capacity-feasible routes and extract edges
    n_savings_starts = 15
    savings_edge_matrix = np.zeros((n, n), dtype=float)
    savings_costs = []
    
    for i in range(n_savings_starts):
        if time.time() > deadline - 80:
            break
        routes = _savings_routes(distance_matrix, demands, capacity, seed=i)
        
        total_cost = 0.0
        for r in range(len(routes)):
            routes[r] = _two_opt_route(routes[r], distance_matrix, demands, capacity)
            for k in range(len(routes[r]) - 1):
                u, v = routes[r][k], routes[r][k+1]
                savings_edge_matrix[u, v] += 1.0
                total_cost += distance_matrix[u, v]
        
        savings_costs.append(total_cost)
    
    # --- 3. Combine signals with Lagrangian-inspired weighting ---
    # Base heuristic: 1/distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Normalize edge frequency matrices
    if np.max(tsp_edge_matrix) > 0:
        tsp_norm = tsp_edge_matrix / np.max(tsp_edge_matrix)
    else:
        tsp_norm = np.zeros_like(tsp_edge_matrix)
    
    if np.max(savings_edge_matrix) > 0:
        savings_norm = savings_edge_matrix / np.max(savings_edge_matrix)
    else:
        savings_norm = np.zeros_like(savings_edge_matrix)
    
    # Weight by solution quality: TSP tours give global structure, savings give feasibility
    # Use soft-min to smoothly combine: weight = exp(-cost/T) / sum(exp(-cost/T))
    # But for simplicity, use fixed weights based on which signal is more reliable
    
    # TSP edges are globally optimal but may violate capacity
    # Savings edges are locally feasible but may miss global structure
    # Lagrangian insight: edges that appear in both are highly attractive
    
    # Create a combined matrix:
    # h = base + alpha * tsp_norm + beta * savings_norm + gamma * (tsp_norm * savings_norm)
    # The product term rewards edges that are both globally good and capacity-feasible
    
    # Search for good parameters
    def _greedy_tour_cost(h, dm, demands, cap):
        """Run a single greedy ant to estimate heuristic quality."""
        current = 0
        remaining_cap = cap
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        
        while True:
            eligible = [j for j in range(1, n) if not visited[j] and demands[j] <= remaining_cap]
            if not eligible:
                total_dist += dm[current, 0]
                current = 0
                remaining_cap = cap
                if visited[1:].all():
                    break
                continue
            
            best_j = max(eligible, key=lambda j: h[current, j])
            total_dist += dm[current, best_j]
            remaining_cap -= demands[best_j]
            visited[best_j] = True
            current = best_j
            
            if visited[1:].all():
                total_dist += dm[current, 0]
                break
        
        return total_dist
    
    best_params = (1.0, 1.0, 2.0)  # alpha, beta, gamma
    best_greedy_cost = float('inf')
    
    # Grid search over parameter space
    alphas = np.logspace(-1, 2, 5)   # 0.1 to 100
    betas = np.logspace(-1, 2, 5)    # 0.1 to 100
    gammas = [0.5, 1.0, 2.0, 5.0]    # Product term weight
    
    for a in alphas:
        if time.time() > deadline - 30:
            break
        for b in betas:
            for g in gammas:
                h_test = base + a * tsp_norm + b * savings_norm + g * (tsp_norm * savings_norm)
                cost = _greedy_tour_cost(h_test, distance_matrix, demands, capacity)
                if cost < best_greedy_cost:
                    best_greedy_cost = cost
                    best_params = (a, b, g)
    
    alpha, beta, gamma = best_params
    heuristic = base + alpha * tsp_norm + beta * savings_norm + gamma * (tsp_norm * savings_norm)
    
    # Ensure positivity and finiteness
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Final check: shape
    if heuristic.shape != distance_matrix.shape:
        heuristic = 1.0 / (distance_matrix + 1e-9)
        
    return heuristic
