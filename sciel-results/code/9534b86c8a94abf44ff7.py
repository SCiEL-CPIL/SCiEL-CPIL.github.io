"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _run_aco(distance_matrix, coordinates, demands, capacity, heuristic, n_ants=20, n_iterations=30, seed=42):
    """Run ACO and return the best (lowest) total route distance found by any ant."""
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    pheromone = np.ones_like(heuristic, dtype=float)
    # Ensure heuristic is positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_overall = float('inf')
    
    for it in range(n_iterations):
        best_iter_cost = float('inf')
        
        for ant in range(n_ants):
            # Ant state
            current = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0
            
            while True:
                # Find eligible customers
                # Unvisited, not depot, demand fits remaining
                eligible_mask = (~visited) & (np.arange(n) > 0)
                eligible_mask &= (demands[:n] <= remaining_cap)
                eligible_indices = np.where(eligible_mask)[0]
                
                if len(eligible_indices) == 0:
                    # Return to depot
                    total_dist += distance_matrix[current, 0]
                    current = 0
                    remaining_cap = capacity
                    # Check if all customers visited
                    if visited[1:].all():
                        break
                    continue
                
                # Compute probabilities
                probs = pheromone[current, eligible_indices] * heuristic[current, eligible_indices]
                probs = np.where(probs > 0, probs, 1e-9)
                probs /= probs.sum()
                
                # Pick next
                idx = rng.choice(len(eligible_indices), p=probs)
                next_node = eligible_indices[idx]
                
                total_dist += distance_matrix[current, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                current = next_node
                
                # If all visited, go back to depot
                if visited[1:].all():
                    total_dist += distance_matrix[current, 0]
                    break
        
        # Track best per iteration
        if total_dist < best_iter_cost:
            best_iter_cost = total_dist
        
        # Track overall best
        if total_dist < best_overall:
            best_overall = total_dist
        
        # Pheromone update
        pheromone *= 0.9
        pheromone += 1.0 / best_iter_cost
    
    return best_overall

def _construct_reference_solution(distance_matrix, demands, capacity, n_customers, max_restarts=20, seed=123):
    """
    Constructs a high-quality reference solution using multi-start Nearest Neighbor 
    followed by 2-opt local search. This provides a 'structural' signal for the heuristic.
    """
    n = n_customers + 1
    rng = np.random.RandomState(seed)
    best_solution_edges = np.zeros((n, n), dtype=float)
    best_cost = float('inf')
    
    def two_opt_improve(route, dist_matrix, demands, capacity):
        """
        Improves a single route using 2-opt.
        Route is a list of nodes [0, c1, c2, ..., 0].
        Returns improved route and cost.
        """
        if len(route) <= 2:
            return route, 0.0
            
        # Calculate initial cost
        cost = 0.0
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i], route[i+1]]
            
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Check if reversing segment [i+1:j] is valid
                    # In CVRP, 2-opt must respect capacity. 
                    # Simple 2-opt on a single route usually maintains capacity if the route was valid,
                    # because the set of nodes in the route doesn't change, only their order.
                    # However, we must ensure the "cut" doesn't violate capacity? 
                    # No, capacity is a constraint on the sum of demands in the route. 
                    # Reordering nodes in a valid route keeps the sum of demands the same.
                    # So standard 2-opt is safe for capacity.
                    
                    # Calculate delta cost
                    # Remove edges (route[i-1], route[i]) and (route[j], route[j+1])
                    # Add edges (route[i-1], route[j]) and (route[i], route[j+1])
                    # Wait, standard 2-opt reverses segment i..j.
                    # Edges removed: (route[i-1], route[i]) and (route[j], route[j+1])
                    # Edges added: (route[i-1], route[j]) and (route[i], route[j+1])
                    
                    u = route[i-1]
                    v = route[i]
                    w = route[j]
                    x = route[j+1]
                    
                    delta = dist_matrix[u, w] + dist_matrix[v, x] - dist_matrix[u, v] - dist_matrix[w, x]
                    
                    if delta < -1e-7:
                        route[i:j+1] = route[i:j+1][::-1]
                        cost += delta
                        improved = True
                        
        return route, cost

    def nearest_neighbor_route(start_node, remaining_customers, dist_matrix, demands, capacity):
        """
        Builds a route starting from start_node using nearest neighbor heuristic,
        respecting capacity.
        """
        route = [start_node]
        current = start_node
        remaining_cap = capacity
        unvisited = set(remaining_customers)
        
        while unvisited:
            # Find nearest unvisited customer that fits
            best_next = -1
            best_dist = float('inf')
            for c in unvisited:
                if demands[c] <= remaining_cap:
                    d = dist_matrix[current, c]
                    if d < best_dist:
                        best_dist = d
                        best_next = c
                        
            if best_next == -1:
                break
                
            route.append(best_next)
            remaining_cap -= demands[best_next]
            unvisited.remove(best_next)
            current = best_next
            
        route.append(0) # Return to depot
        return route

    def evaluate_solution(routes):
        """
        Takes a list of routes (each a list of nodes starting/ending with 0)
        and returns the total cost and an edge matrix.
        """
        total_cost = 0.0
        edge_matrix = np.zeros((n, n), dtype=float)
        
        for route in routes:
            for i in range(len(route) - 1):
                u = route[i]
                v = route[i+1]
                total_cost += distance_matrix[u, v]
                edge_matrix[u, v] = 1.0
                
        return total_cost, edge_matrix

    customers = list(range(1, n))
    
    for restart in range(max_restarts):
        # Shuffle customers to create different starting points for NN
        shuffled_customers = customers[:]
        rng.shuffle(shuffled_customers)
        
        routes = []
        remaining_customers = set(shuffled_customers)
        
        # Build routes using NN
        # We start a new route whenever we can't fit more customers or we run out.
        # A simple strategy: start with the first remaining customer, build a route.
        # Then take the next remaining customer, build a route.
        # This is a "partitioning" NN.
        
        i = 0
        while i < len(shuffled_customers):
            start = shuffled_customers[i]
            # Build route starting with 'start'
            route = nearest_neighbor_route(start, remaining_customers, distance_matrix, demands, capacity)
            if len(route) > 2: # If route has customers
                routes.append(route)
                # Remove used customers from remaining
                for node in route[1:-1]:
                    remaining_customers.remove(node)
                # Advance index past used customers
                # Since we removed them, we just need to find the next one in shuffled list that is still remaining
                # Actually, it's easier to just iterate through shuffled_customers and check if they are in remaining_customers
            i += 1
            
        # Check if any customers remain (shouldn't happen if capacity is enough, but just in case)
        if remaining_customers:
             # Force add them to existing routes or create new ones
             # For simplicity in this fast heuristic, we assume capacity allows it or we just add them to the last route
             # This might violate capacity, so we should be careful.
             # Given capacity 50 and max demand 9, and 50 customers, avg demand ~5.
             # Max customers per route ~ 10. Total routes ~ 5.
             # NN should handle this.
             pass

        # Apply 2-opt to each route
        improved_routes = []
        for route in routes:
            imp_route, imp_cost = two_opt_improve(route, distance_matrix, demands, capacity)
            improved_routes.append(imp_route)
            
        # Evaluate
        cost, edge_mat = evaluate_solution(improved_routes)
        
        if cost < best_cost:
            best_cost = cost
            best_solution_edges = edge_mat

    return best_solution_edges

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 300 # 300s limit, leave margin
    
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # 1. Generate Reference Solution
    ref_edges = _construct_reference_solution(distance_matrix, demands, capacity, n-1, max_restarts=50, seed=42)
    
    # 2. Prepare Base Heuristic Components
    # Component A: Distance based (1/d)
    dist_safe = distance_matrix + 1e-9
    h_dist = 1.0 / dist_safe
    # Normalize h_dist to [0, 1] range roughly for easier combination
    h_dist_max = np.max(h_dist)
    h_dist_norm = h_dist / h_dist_max
    
    # Component B: Reference Solution Edges
    # h_ref is 1 where the reference solution has an edge, 0 otherwise
    h_ref = ref_edges.copy()
    
    # 3. Optimize Heuristic Weights
    # Heuristic H = w1 * h_dist_norm + w2 * h_ref
    # We want to find w1, w2 such that ACO performance is best.
    # Since H is multiplied by pheromone (which starts at 1), the relative scale matters.
    # We can fix the scale by normalizing H such that max(H) = 1.
    
    best_score = float('inf')
    best_w1, best_w2 = 1.0, 0.0 # Default to pure distance
    
    # Grid search over weights
    # w1 in [0.1, 1.0], w2 in [0.0, 2.0]
    # Note: If w2 is high, the ACO will strongly prefer the reference edges.
    # If w1 is high, it prefers short edges.
    
    w1_candidates = [0.1, 0.3, 0.5, 1.0]
    w2_candidates = [0.0, 0.5, 1.0, 2.0, 3.0]
    
    for w1 in w1_candidates:
        if time.time() > deadline - 10:
            break
        for w2 in w2_candidates:
            if time.time() > deadline - 10:
                break
                
            # Construct heuristic
            h_candidate = w1 * h_dist_norm + w2 * h_ref
            
            # Normalize to max 1
            max_h = np.max(h_candidate)
            if max_h > 0:
                h_candidate = h_candidate / max_h
            else:
                h_candidate = np.ones_like(h_candidate)
                
            # Ensure positive
            h_candidate = np.where(h_candidate > 0, h_candidate, 1e-9)
            
            try:
                score = _run_aco(distance_matrix, coordinates, demands, capacity, h_candidate, seed=42)
                if score < best_score:
                    best_score = score
                    best_w1, best_w2 = w1, w2
            except Exception:
                continue

    # 4. Construct Final Heuristic
    final_heuristic = best_w1 * h_dist_norm + best_w2 * h_ref
    max_h = np.max(final_heuristic)
    if max_h > 0:
        final_heuristic = final_heuristic / max_h
    else:
        final_heuristic = np.ones_like(final_heuristic)
        
    final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    # Safety check
    if not np.all(np.isfinite(final_heuristic)):
        final_heuristic = 1.0 / (distance_matrix + 1e-9)
        final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
        
    return final_heuristic
# EVOLVE-BLOCK-END
