"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _run_aco(distance_matrix, demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """Run ACO and return the best (lowest) total route distance found by any ant."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones_like(heuristic, dtype=float)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_overall = float('inf')
    
    for it in range(n_iterations):
        best_iter_cost = float('inf')
        
        for ant in range(n_ants):
            current = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                eligible_mask = (~visited) & (np.arange(n) > 0)
                eligible_mask &= (demands[:n] <= remaining_cap)
                eligible_indices = np.where(eligible_mask)[0]
                
                if len(eligible_indices) == 0:
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    if visited[1:].all():
                        break
                    continue
                
                probs = pheromone[current, eligible_indices] * heuristic[current, eligible_indices]
                probs = np.where(probs > 0, probs, 1e-9)
                probs /= probs.sum()
                
                idx = rng.choice(len(eligible_indices), p=probs)
                next_node = eligible_indices[idx]
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                
                if visited[1:].all():
                    total_dist += distance_matrix[current, 0]
                    break
        
        if total_dist < best_iter_cost:
            best_iter_cost = total_dist
        
        if total_dist < best_overall:
            best_overall = total_dist
        
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
    
    return best_overall

def _solve_cvrp_savings(distance_matrix, demands, capacity, n_customers, max_restarts=30, seed=42):
    """
    Solve CVRP using Clarke-Wright Savings with multi-start and local search.
    Returns the best solution's edge matrix and cost.
    """
    n = n_customers + 1
    rng = np.random.RandomState(seed)
    best_cost = float('inf')
    best_edge_matrix = np.zeros((n, n), dtype=float)
    
    # Precompute savings
    savings = np.zeros((n_customers, n_customers))
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci, cj = i + 1, j + 1
            s = distance_matrix[0, ci] + distance_matrix[0, cj] - distance_matrix[ci, cj]
            savings[i, j] = s
            savings[j, i] = s
    
    # Sort savings in descending order
    savings_pairs = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            savings_pairs.append((savings[i, j], i, j))
    savings_pairs.sort(reverse=True)
    
    def merge_routes(routes, i, j, demands, capacity):
        """Merge route containing customer i+1 with route containing customer j+1."""
        route_i = None
        route_j = None
        idx_i = None
        idx_j = None
        
        for r_idx, route in enumerate(routes):
            if i + 1 in route:
                route_i = r_idx
                idx_i = route.index(i + 1)
            if j + 1 in route:
                route_j = r_idx
                idx_j = route.index(j + 1)
        
        if route_i is None or route_j is None or route_i == route_j:
            return False
        
        # Check capacity
        total_demand = sum(demands[node] for node in routes[route_i]) + sum(demands[node] for node in routes[route_j])
        if total_demand > capacity:
            return False
        
        # Find endpoints
        # Route i: ... i+1 ...
        # Route j: ... j+1 ...
        # We need to merge at endpoints
        
        ri = routes[route_i]
        rj = routes[route_j]
        
        # Find positions of i+1 and j+1
        # The merge should be at the endpoints of the routes
        
        # Check if i+1 is at an endpoint of ri
        if ri[0] == i + 1:
            pos_i = 0
        elif ri[-1] == i + 1:
            pos_i = len(ri) - 1
        else:
            # i+1 is not at an endpoint, can't merge easily
            return False
        
        if rj[0] == j + 1:
            pos_j = 0
        elif rj[-1] == j + 1:
            pos_j = len(rj) - 1
        else:
            return False
        
        # Merge
        # Reverse route if needed so that the merge point is at the end
        if pos_i == 0:
            ri = ri[::-1]  # Now i+1 is at the end
        if pos_j == 0:
            rj = rj[::-1]  # Now j+1 is at the end
        
        # Now ri ends with i+1, rj ends with j+1
        # Merge: ri[:-1] + rj (since rj starts with depot, we skip it)
        new_route = ri[:-1] + rj
        
        routes[route_i] = new_route
        routes[route_j] = None
        
        return True
    
    def two_opt_route(route, distance_matrix, max_passes=10):
        """Apply 2-opt to a single route."""
        if len(route) <= 2:
            return route
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    u = route[i - 1]
                    v = route[i]
                    w = route[j]
                    x = route[j + 1]
                    
                    delta = distance_matrix[u, w] + distance_matrix[v, x] - distance_matrix[u, v] - distance_matrix[w, x]
                    
                    if delta < -1e-10:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True
        
        return route
    
    def or_opt_route(route, distance_matrix, max_passes=5):
        """Apply Or-opt to a single route."""
        if len(route) <= 3:
            return route
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for seg_len in [1, 2, 3]:
                if len(route) <= seg_len + 2:
                    continue
                for i in range(1, len(route) - seg_len - 1):
                    segment = route[i:i + seg_len]
                    route_removed = route[:i] + route[i + seg_len:]
                    
                    for j in range(1, len(route_removed) - 1):
                        if j == i or j == i + 1:
                            continue
                        
                        new_route = route_removed[:j] + segment + route_removed[j:]
                        
                        # Calculate cost change
                        old_cost = distance_matrix[route[i - 1], route[i]] + distance_matrix[route[i + seg_len - 1], route[i + seg_len]]
                        new_cost = distance_matrix[route_removed[j - 1], segment[0]] + distance_matrix[segment[-1], route_removed[j]]
                        
                        if new_cost < old_cost - 1e-10:
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return route
    
    def evaluate_routes(routes, distance_matrix):
        """Evaluate total cost of routes and build edge matrix."""
        total_cost = 0.0
        edge_matrix = np.zeros((n, n), dtype=float)
        
        for route in routes:
            if route is None:
                continue
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                total_cost += distance_matrix[u, v]
                edge_matrix[u, v] += 1.0
        
        return total_cost, edge_matrix
    
    customers = list(range(1, n))
    
    for restart in range(max_restarts):
        # Initialize: each customer is its own route
        routes = [[0, c, 0] for c in customers]
        
        # Apply savings
        for s, i, j in savings_pairs:
            if s <= 0:
                break
            if merge_routes(routes, i, j, demands, capacity):
                continue
        
        # Remove None routes
        routes = [r for r in routes if r is not None]
        
        # Apply local search to each route
        for idx, route in enumerate(routes):
            route = two_opt_route(route, distance_matrix)
            route = or_opt_route(route, distance_matrix)
            routes[idx] = route
        
        # Evaluate
        cost, edge_mat = evaluate_routes(routes, distance_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_edge_matrix = edge_mat.copy()
    
    return best_edge_matrix, best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, leave margin
    
    n = distance_matrix.shape[0]
    n_customers = n - 1
    
    # 1. Solve CVRP using Clarke-Wright Savings with local search
    ref_edges, ref_cost = _solve_cvrp_savings(distance_matrix, demands, capacity, n_customers, max_restarts=50, seed=42)
    
    # 2. Compute distance-based heuristic
    dist_safe = distance_matrix + 1e-9
    h_dist = 1.0 / dist_safe
    h_dist_max = np.max(h_dist)
    if h_dist_max > 0:
        h_dist_norm = h_dist / h_dist_max
    else:
        h_dist_norm = np.ones_like(h_dist)
    
    # 3. Combine: reference edges + distance
    # Use a blend that emphasizes reference solution edges
    h_ref = ref_edges.copy()
    h_ref_max = np.max(h_ref)
    if h_ref_max > 0:
        h_ref_norm = h_ref / h_ref_max
    else:
        h_ref_norm = np.zeros_like(h_ref)
    
    # Blend: 60% reference, 40% distance
    final_heuristic = 0.6 * h_ref_norm + 0.4 * h_dist_norm
    
    # Ensure positive
    final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    # Normalize to max 1
    max_h = np.max(final_heuristic)
    if max_h > 0:
        final_heuristic = final_heuristic / max_h
    
    # Safety check
    if not np.all(np.isfinite(final_heuristic)):
        final_heuristic = 1.0 / (distance_matrix + 1e-9)
        final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
