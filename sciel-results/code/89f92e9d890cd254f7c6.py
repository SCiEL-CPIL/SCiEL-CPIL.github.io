# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Strategy:
    1. Base heuristic: Inverse distance (1/d).
    2. Structural boost: Use a Local Search (2-Opt) refined Nearest Neighbor 
       heuristic to identify high-quality route segments.
    3. Count edge frequencies in these refined solutions and boost the heuristic.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        new_demands = np.zeros(n, dtype=np.float64)
        min_len = min(demands.shape[0], n)
        new_demands[:min_len] = demands[:min_len]
        demands = new_demands

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # 1. Base Heuristic: Inverse distance
    base_heur = 1.0 / dist_safe
    # Set depot self-loop to 0 or small value, though ACO usually handles this
    np.fill_diagonal(base_heur, 0.0)

    # 2. Local Search Heuristic
    # We will generate several feasible solutions using Nearest Neighbor + 2-Opt
    # and count edge frequencies.
    
    edge_count = np.zeros((n, n), dtype=np.float64)
    
    # Number of restarts for the construction heuristic
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # --- Construct Initial Solution using Nearest Neighbor ---
        # We simulate multiple vehicles (routes) starting from depot
        # Since we don't know the number of vehicles, we just keep adding routes
        # until all customers are visited.
        
        unvisited = set(range(1, n))
        routes = []
        
        while unvisited:
            if time.time() > deadline - 10.0:
                break
                
            # Start a new route from depot
            current_route = [0]
            current_load = 0.0
            current_node = 0
            
            while True:
                # Find nearest unvisited customer that fits in capacity
                best_next = -1
                best_dist = np.inf
                
                # Optimization: If many unvisited, we can vectorize, but n=50 is small
                # so a loop is fine and allows early termination if capacity is tight.
                # However, to be safe and fast, let's just iterate.
                candidates = []
                for node in unvisited:
                    if current_load + demands[node] <= capacity:
                        d = distance_matrix[current_node, node]
                        if d < best_dist:
                            best_dist = d
                            best_next = node
                
                if best_next == -1:
                    # No customer fits, return to depot
                    break
                
                # Move to best_next
                current_route.append(best_next)
                current_load += demands[best_next]
                unvisited.remove(best_next)
                current_node = best_next
                
                if not unvisited:
                    break
            
            # Return to depot
            current_route.append(0)
            routes.append(current_route)
            
        # --- Improve Solution using 2-Opt ---
        # We apply 2-Opt to each route independently.
        # Note: Standard 2-Opt reverses a segment. In a route [0, A, B, C, 0],
        # reversing [A, B] gives [0, B, A, C, 0].
        
        for route in routes:
            # If route is too short, skip
            if len(route) < 4:
                continue
                
            improved = True
            max_iter = 100 # Limit 2-Opt iterations to prevent hanging
            iter_count = 0
            
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                if time.time() > deadline - 10.0:
                    break
                    
                L = len(route)
                # Consider edges between route[1]..route[L-2]
                # Standard 2-Opt on a cycle: remove (i, i+1) and (j, j+1), add (i, j) and (i+1, j+1)
                # Here indices are in the route list.
                # We must be careful with the depot (index 0 and L-1).
                
                # Iterate over pairs of edges
                for i in range(1, L - 2):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i + 1, L - 1):
                        # Skip if edges are adjacent (i+1 == j)
                        if i + 1 == j:
                            continue
                        
                        # Nodes involved
                        n1 = route[i-1]
                        n2 = route[i]
                        n3 = route[j]
                        n4 = route[j+1]
                        
                        # Calculate cost change
                        # Old edges: (n1, n2) and (n3, n4)
                        # New edges: (n1, n3) and (n2, n4)
                        # Note: The segment between n2 and n3 is reversed.
                        
                        old_cost = distance_matrix[n1, n2] + distance_matrix[n3, n4]
                        new_cost = distance_matrix[n1, n3] + distance_matrix[n2, n4]
                        
                        if new_cost < old_cost - 1e-10:
                            # Perform reversal
                            # Reverse route[i] to route[j]
                            route[i:j+1] = route[i:j+1][::-1]
                            improved = True
                            if time.time() > deadline - 10.0:
                                break

        # --- Count Edges from Final Routes ---
        for route in routes:
            for k in range(len(route) - 1):
                u = route[k]
                v = route[k+1]
                edge_count[u, v] += 1.0
                edge_count[v, u] += 1.0

    # 3. Apply Boost
    # Combine base heuristic with edge frequency.
    # We want edges that are frequent in good solutions to be more attractive.
    # H[i,j] = Base[i,j] * (1 + alpha * Count[i,j])
    
    # Normalize counts to be on a similar scale
    max_count = np.max(edge_count)
    if max_count > 0:
        # Scale counts to [0, 1] roughly, then boost
        # A factor of 10 seems reasonable to make frequent edges significantly better
        boost_matrix = 1.0 + 10.0 * (edge_count / max_count)
        base_heur = base_heur * boost_matrix

    # 4. Final Sanitization
    # Ensure all entries are strictly positive and finite
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9
    # Ensure diagonal is small but positive? The contract says non-positive becomes 1e-9.
    # The ACO loop usually skips visited nodes, so diagonal might not matter much, 
    # but let's keep it consistent.
    np.fill_diagonal(base_heur, 1e-9)

    return base_heur
# EVOLVE-BLOCK-END
