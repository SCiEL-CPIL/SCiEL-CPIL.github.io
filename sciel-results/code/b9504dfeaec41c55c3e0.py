"""Lagrangian Relaxation-based edge weight heuristic for CVRP ant colony."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Lagrangian Relaxation approach.
    
    The idea: solve a relaxed problem where we ignore capacity constraints but add
    a penalty term for load violations. Edges that are part of near-optimal tours
    under this relaxed formulation get higher attractiveness.
    
    We use a dual ascent approach:
    1. Start with a baseline distance-based heuristic.
    2. Iteratively adjust Lagrange multipliers for capacity violations.
    3. The final edge weights reflect which edges are "cheap" both in distance
       and in terms of load efficiency.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=float)
    
    # Ensure demands[0] = 0 (depot)
    demands[0] = 0.0
    
    # Baseline: inverse distance
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Lagrangian relaxation: we want to identify edges that are good
    # in terms of both distance and load efficiency.
    
    # Step 1: Compute a "load efficiency" score for each edge.
    # An edge (i, j) is more efficient if it helps keep routes within capacity.
    # We estimate this by looking at the cumulative demand along potential paths.
    
    # For simplicity, we use a dual-variable approach:
    # Let lambda be a scalar multiplier for capacity violation.
    # The relaxed cost of a route is: sum(dist) + lambda * max(0, total_demand - capacity)
    
    # We approximate by computing for each edge how much it "contributes" to
    # exceeding capacity, weighted by the demand of the destination node.
    
    # Edge load contribution: entering node j adds demand[j] to the current load.
    # If we're already near capacity, this edge is more costly in the relaxed sense.
    
    # Heuristic: for each edge (i, j), the "relaxed cost" is:
    #   dist[i,j] + lambda * demand[j] * (1 - remaining_capacity_fraction)
    
    # We estimate remaining_capacity_fraction as a function of how "full" a typical
    # route is. Since we don't know the actual routes, we use a global estimate.
    
    # Average demand per customer
    avg_demand = np.mean(demands[1:]) if n > 1 else 1.0
    
    # Estimated number of customers per route
    est_customers_per_route = max(1, int(capacity / max(avg_demand, 1e-9)))
    
    # For each edge (i, j), compute a load-aware cost:
    # The key insight: edges that lead to high-demand nodes should be penalized
    # more when the route is likely to be near capacity.
    
    # We use a simple model: the probability that adding demand[j] would exceed
    # capacity is proportional to demand[j] / capacity.
    
    # Load penalty for edge (i, j): lambda * demand[j] / capacity
    # This is a crude but effective approximation.
    
    # We run a few iterations of dual ascent to find a good lambda.
    
    lambda_val = 0.0
    best_heuristic = base_heuristic.copy()
    
    # Try several lambda values and pick the one that gives the best "separation"
    # between good and bad edges.
    
    # Actually, let's use a more principled approach:
    # Compute the relaxed TSP tour (ignoring capacity) using nearest neighbor
    # with load-aware costs, then use the edges in that tour as a guide.
    
    # Build load-aware distance matrix for several lambda values
    lambda_candidates = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    # For each lambda, compute a relaxed cost matrix:
    # relaxed_cost[i, j] = dist[i, j] + lambda * demand[j] / capacity
    # (for j != 0; for j = 0, no load is added)
    
    best_score = np.inf
    best_heur = base_heuristic.copy()
    
    for lam in lambda_candidates:
        # Relaxed cost matrix
        relaxed_cost = distance_matrix.copy()
        # Add load penalty: entering node j adds demand[j]
        # Create a matrix where each entry (i, j) gets lambda * demand[j] / capacity
        load_penalty = np.outer(np.ones(n), demands / max(capacity, 1e-9)) * lam
        # No penalty for returning to depot (j=0)
        load_penalty[:, 0] = 0.0
        relaxed_cost = relaxed_cost + load_penalty
        
        # Ensure positive
        relaxed_cost = np.maximum(relaxed_cost, 1e-9)
        
        # Inverse as heuristic
        heur_candidate = 1.0 / relaxed_cost
        
        # Score: we want edges that are short AND don't add too much load
        # A simple score: sum of heuristics for edges in a greedy tour
        # Let's compute a greedy nearest-neighbor tour using relaxed_cost
        # and sum the original distances of edges in that tour.
        
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        total_dist = 0.0
        total_load = 0.0
        route_edges = []
        
        for _ in range(n - 1):
            # Find nearest unvisited node using relaxed_cost, but only if it fits
            best_j = -1
            best_cost = np.inf
            for j in range(1, n):
                if not visited[j]:
                    if total_load + demands[j] <= capacity:
                        if relaxed_cost[current, j] < best_cost:
                            best_cost = relaxed_cost[current, j]
                            best_j = j
            if best_j == -1:
                # No node fits, go back to depot
                total_dist += distance_matrix[current, 0]
                current = 0
                total_load = 0.0
                # Try again
                for j in range(1, n):
                    if not visited[j]:
                        if total_load + demands[j] <= capacity:
                            if relaxed_cost[current, j] < best_cost:
                                best_cost = relaxed_cost[current, j]
                                best_j = j
                if best_j == -1:
                    break
            visited[best_j] = True
            total_dist += distance_matrix[current, best_j]
            total_load += demands[best_j]
            current = best_j
            route_edges.append((current - 1 if current > 0 else 0, best_j))  # rough
        
        # Final return to depot
        total_dist += distance_matrix[current, 0]
        
        # Score is total_dist (lower is better)
        if total_dist < best_score:
            best_score = total_dist
            best_heur = heur_candidate.copy()
    
    # Combine: use the best lambda's heuristic, but also blend with base
    # to ensure numerical stability
    result = 0.5 * base_heuristic + 0.5 * best_heur
    
    # Ensure all values are positive
    result = np.maximum(result, 1e-9)
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
