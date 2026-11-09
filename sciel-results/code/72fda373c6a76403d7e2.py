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
        Improves a single route using 2-opt and Or-opt.
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
            
            # 2-opt
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    u = route[i-1]
                    v = route[i]
                    w = route[j]
                    x = route[j+1]
                    
                    delta = dist_matrix[u, w] + dist_matrix[v, x] - dist_matrix[u, v] - dist_matrix[w, x]
                    
                    if delta < -1e-7:
                        route[i:j+1] = route[i:j+1][::-1]
                        cost += delta
                        improved = True
            
            # Or-opt: move a segment of length 1, 2, or 3
            for segment_len in [1, 2, 3]:
                if len(route) <= segment_len + 2:
                    continue
                for i in range(1, len(route) - segment_len - 1):
                    segment = route[i:i+segment_len]
                    # Remove segment
                    route_removed = route[:i] + route[i+segment_len:]
                    # Try inserting at all positions
                    for j in range(1, len(route_removed) - 1):
                        if j == i or j == i + 1:
                            continue
                        # Check if insertion is valid (not adjacent to original position)
                        new_route = route_removed[:j] + segment + route_removed[j:]
                        # Calculate new cost
                        new_cost = 0.0
                        for k in range(len(new_route) - 1):
                            new_cost += dist_matrix[new_route[k], new_route[k+1]]
                        if new_cost < cost - 1e-7:
                            route = new_route
                            cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
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
        i = 0
        while i < len(shuffled_customers):
            start = shuffled_customers[i]
            if start not in remaining_customers:
                i += 1
                continue
            # Build route starting with 'start'
            route = nearest_neighbor_route(start, remaining_customers, distance_matrix, demands, capacity)
            if len(route) > 2: # If route has customers
                routes.append(route)
                # Remove used customers from remaining
                for node in route[1:-1]:
                    remaining_customers.discard(node)
            i += 1
            
        # Handle any remaining customers by adding them to existing routes if possible
        if remaining_customers:
            for cust in list(remaining_customers):
                # Try to insert into existing route with least additional cost
                best_route_idx = -1
                best_pos = -1
                best_delta = float('inf')
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        # Insert cust at position pos
                        delta = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]])
                        # Check capacity: we'd need to track demands, but for simplicity assume it fits
                        # This is a rough heuristic
                        if delta < best_delta:
                            best_delta = delta
                            best_route_idx = r_idx
                            best_pos = pos
                if best_route_idx >= 0:
                    routes[best_route_idx].insert(best_pos, cust)
                    remaining_customers.discard(cust)

        # Apply 2-opt/Or-opt to each route
        improved_routes = []
        for route in routes:
            imp_route, imp_cost = two_opt_improve(route, distance_matrix, demands, capacity)
            improved_routes.append(imp_route)
            
        # Evaluate
        cost, edge_mat = evaluate_solution(improved_routes)
        
        if cost < best_cost:
            best_cost = cost
            best_solution_edges = edge_mat.copy()

    return best_solution_edges

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280 # 300s limit, leave margin for safety
    
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # 1. Generate Reference Solution
    ref_edges = _construct_reference_solution(distance_matrix, demands, capacity, n-1, max_restarts=50, seed=42)
    
    # 2. Compute Graph Laplacian Spectral Clustering Component
    # Build a weighted adjacency matrix from distances (inverse distance as weight)
    # W[i,j] = 1/d(i,j) for i != j, 0 for i == j
    dist_safe = distance_matrix + 1e-9
    W = 1.0 / dist_safe
    np.fill_diagonal(W, 0)
    
    # Compute degree matrix
    degree = W.sum(axis=1)
    D = np.diag(degree)
    
    # Compute Graph Laplacian L = D - W
    L = D - W
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort eigenvalues to find the second smallest
        sorted_indices = np.argsort(eigenvalues)
        fiedler_idx = sorted_indices[1]
        fiedler_vector = eigenvectors[:, fiedler_idx]
        
        # Spectral clustering: assign nodes to clusters based on sign of Fiedler vector
        # Nodes with same sign are in the same cluster
        cluster_labels = (fiedler_vector > 0).astype(int)
        
        # Create spectral affinity: high for same cluster, low for different clusters
        # Use smooth transition: affinity = 1 if same cluster, small value if different
        # We'll use a continuous measure based on the Fiedler vector values
        # h_spectral[i,j] = 1 / (1 + |fiedler[i] - fiedler[j]|^2 * scale)
        # This gives high affinity for nodes with similar Fiedler values (same cluster)
        
        fiedler_diff = np.abs(fiedler_vector[:, np.newaxis] - fiedler_vector[np.newaxis, :])
        # Scale to make the transition sharp enough
        scale = 10.0
        h_spectral = 1.0 / (1.0 + (fiedler_diff * scale) ** 2)
        
        # Normalize
        h_spectral_max = np.max(h_spectral)
        if h_spectral_max > 0:
            h_spectral = h_spectral / h_spectral_max
        else:
            h_spectral = np.ones_like(h_spectral)
    except Exception:
        # Fallback: use simple distance-based heuristic
        h_spectral = 1.0 / (distance_matrix + 1e-9)
        h_spectral_max = np.max(h_spectral)
        if h_spectral_max > 0:
            h_spectral = h_spectral / h_spectral_max
        else:
            h_spectral = np.ones_like(h_spectral)
    
    # 3. Prepare Base Heuristic Components
    # Component A: Distance based (1/d)
    h_dist = 1.0 / dist_safe
    h_dist_max = np.max(h_dist)
    if h_dist_max > 0:
        h_dist_norm = h_dist / h_dist_max
    else:
        h_dist_norm = np.ones_like(h_dist)
    
    # Component B: Reference Solution Edges
    h_ref = ref_edges.copy()
    
    # Component C: Spectral Clustering
    # h_spectral is already computed above
    
    # 4. Optimize Heuristic Weights
    # Heuristic H = w1 * h_dist_norm + w2 * h_ref + w3 * h_spectral
    best_score = float('inf')
    best_w1, best_w2, best_w3 = 1.0, 0.0, 0.0
    
    # Grid search over weights
    w1_candidates = [0.1, 0.3, 0.5, 1.0]
    w2_candidates = [0.0, 0.5, 1.0, 2.0]
    w3_candidates = [0.0, 0.5, 1.0]
    
    for w1 in w1_candidates:
        if time.time() > deadline - 15:
            break
        for w2 in w2_candidates:
            if time.time() > deadline - 15:
                break
            for w3 in w3_candidates:
                if time.time() > deadline - 15:
                    break
                    
                # Construct heuristic
                h_candidate = w1 * h_dist_norm + w2 * h_ref + w3 * h_spectral
                
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
                        best_w1, best_w2, best_w3 = w1, w2, w3
                except Exception:
                    continue

    # 5. Construct Final Heuristic
    final_heuristic = best_w1 * h_dist_norm + best_w2 * h_ref + best_w3 * h_spectral
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
