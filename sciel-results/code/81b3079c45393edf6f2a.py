"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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

def _two_opt_improve(routes, distance_matrix, demands, capacity):
    """
    Improves a set of routes using 2-Opt within each route.
    routes: list of lists of nodes (excluding depot, assuming start/end at depot is implicit or handled)
    Actually, let's represent routes as sequences including depot? 
    Standard 2-Opt works on the sequence. 
    Route: [0, c1, c2, ..., ck, 0]
    We will try to reverse segments.
    """
    improved = True
    while improved:
        improved = False
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            # route is [0, ..., 0]
            # Try all pairs (i, j) where 0 < i < j < len(route)-1
            # Reversing route[i:j+1]
            n_route = len(route)
            for i in range(1, n_route - 2):
                for j in range(i + 1, n_route - 1):
                    # Check if reversing route[i:j+1] improves distance
                    # Current edges: (route[i-1], route[i]) and (route[j], route[j+1])
                    # New edges: (route[i-1], route[j]) and (route[i], route[j+1])
                    # Note: route[0] is 0, route[-1] is 0.
                    # If i=1, route[i-1] is depot.
                    # If j=n_route-2, route[j+1] is depot.
                    
                    u = route[i-1]
                    v = route[i]
                    w = route[j]
                    x = route[j+1]
                    
                    old_dist = distance_matrix[u, v] + distance_matrix[w, x]
                    new_dist = distance_matrix[u, w] + distance_matrix[v, x]
                    
                    if new_dist < old_dist - 1e-10:
                        # Perform reversal
                        routes[r_idx][i:j+1] = routes[r_idx][i:j+1][::-1]
                        improved = True
    return routes

def _nearest_neighbor_route(start, unvisited, distance_matrix, demands, capacity):
    """
    Constructs a single route starting from depot (0) using Nearest Neighbor.
    Returns list of nodes [0, c1, ..., ck, 0]
    """
    current = 0
    remaining_cap = capacity
    route = [0]
    
    while unvisited:
        # Find nearest eligible
        best_next = -1
        best_dist = float('inf')
        for node in unvisited:
            if demands[node] <= remaining_cap:
                d = distance_matrix[current, node]
                if d < best_dist:
                    best_dist = d
                    best_next = node
        
        if best_next == -1:
            break
            
        route.append(best_next)
        remaining_cap -= demands[best_next]
        unvisited.remove(best_next)
        current = best_next
        
    route.append(0)
    return route, unvisited

def _construct_reference_solutions(distance_matrix, coordinates, demands, capacity, n_starts=10, seed=123):
    """
    Generates multiple reference solutions using Multi-Start Nearest Neighbor + 2-Opt.
    Returns a list of edge sets (as boolean matrices or lists of tuples) and their costs.
    """
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    rng = random.Random(seed)
    
    solutions = [] # List of (cost, edge_matrix)
    
    for start_idx in range(n_starts):
        # Randomize the order of customers for the first selection?
        # Standard NN picks the closest. To diversify, we can randomly pick the first customer 
        # or perturb distances slightly.
        # Let's use a simple perturbation: add small random noise to distances for this start.
        # Or just shuffle the initial candidate pool.
        
        unvisited = list(customers)
        rng.shuffle(unvisited)
        
        routes = []
        current_unvisited = list(unvisited)
        
        # Start from depot
        while current_unvisited:
            # Pick a random starting customer from the remaining ones?
            # Or just NN from depot.
            # To ensure diversity, let's pick a random customer from the remaining as the first stop of this route.
            # Then continue with NN from that customer.
            
            # Actually, standard multi-start NN often just shuffles the priority or picks random start.
            # Let's pick a random customer to start the route.
            start_node = rng.choice(current_unvisited)
            
            # Build route starting with start_node
            route_nodes = [0, start_node]
            remaining_cap = capacity - demands[start_node]
            current = start_node
            current_unvisited.remove(start_node)
            
            while current_unvisited:
                best_next = -1
                best_dist = float('inf')
                for node in current_unvisited:
                    if demands[node] <= remaining_cap:
                        d = distance_matrix[current, node]
                        if d < best_dist:
                            best_dist = d
                            best_next = node
                if best_next == -1:
                    break
                route_nodes.append(best_next)
                remaining_cap -= demands[best_next]
                current = best_next
                current_unvisited.remove(best_next)
                
            route_nodes.append(0)
            routes.append(route_nodes)
            
        # Apply 2-Opt improvement
        routes = _two_opt_improve(routes, distance_matrix, demands, capacity)
        
        # Calculate cost and extract edges
        total_cost = 0.0
        edge_matrix = np.zeros((n, n), dtype=bool)
        
        for route in routes:
            for k in range(len(route) - 1):
                u = route[k]
                v = route[k+1]
                total_cost += distance_matrix[u, v]
                edge_matrix[u, v] = True
                # Since distance is symmetric, and ACO traverses directed, 
                # we should probably mark both u->v and v->u? 
                # In a valid route 0->A->B->0, the edges are (0,A), (A,B), (B,0).
                # The reverse edges (A,0), (B,A), (0,B) are not in this specific route.
                # However, if another ant traverses in reverse, it might need them.
                # But CVRP routes are directed in terms of capacity accumulation? 
                # No, capacity is just a sum. The path is a sequence.
                # The heuristic h[i,j] guides i->j.
                # If the optimal solution uses edge A->B, we want h[A,B] to be high.
                # If it uses B->A (in a different route or orientation), we want h[B,A] to be high.
                # Since we don't know the orientation of the "true" optimal, and distance is symmetric,
                # marking both directions is a safe "undirected" heuristic assumption.
                # However, the ACO simulation is directed.
                # Let's mark only the directed edges found in the constructed solution.
                # This is more precise.
        
        solutions.append((total_cost, edge_matrix))
        
    return solutions

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s buffer for safety
    
    n = distance_matrix.shape[0]
    
    # 1. Generate reference solutions
    # n_starts depends on time budget. 10 starts is fast for n=50.
    # 2-Opt is O(n^2) per route, routes are short. Total is very fast.
    try:
        solutions = _construct_reference_solutions(distance_matrix, coordinates, demands, capacity, n_starts=20, seed=42)
    except Exception as e:
        # Fallback if construction fails
        solutions = []
    
    # 2. Construct heuristic matrix
    # Strategy: 
    # Base: 1/distance (standard)
    # Boost: If an edge (i,j) appears in any of the top K solutions, boost it significantly.
    # We can weight the boost by the quality (1/cost) of the solution it came from.
    
    # Sort solutions by cost
    solutions.sort(key=lambda x: x[0])
    
    # Use the best 5 solutions to define the "consensus" edges
    top_solutions = solutions[:5]
    
    # Initialize heuristic with 1/d
    dm_safe = distance_matrix + 1e-9
    heuristic = 1.0 / dm_safe
    # Normalize to [0, 1] for base
    max_base = np.max(heuristic)
    if max_base > 0:
        heuristic = heuristic / max_base
        
    # Boost edges from top solutions
    # We want to strongly attract edges that are part of good solutions.
    # Let's set h[i,j] = max(h[i,j], boost_factor) for all edges in top solutions.
    # Or additive?
    # If we set it to a high constant, we create a binary "good/bad" distinction.
    # If we add to it, we preserve some distance information.
    
    # Let's try: h[i,j] = base[i,j] + sum( weight_k * indicator(i,j in solution_k) )
    # Where weight_k is high for the best solution.
    
    boost_matrix = np.zeros((n, n), dtype=float)
    for idx, (cost, edge_mat) in enumerate(top_solutions):
        # Weight decays with rank
        weight = 1.0 / (idx + 1)
        boost_matrix += weight * edge_mat.astype(float)
        
    # Normalize boost_matrix to be comparable to base
    # Max possible boost is sum(1/k) for k=1..5 ~ 2.28
    # Base is in [0, 1]
    # Let's scale boost so that the max boost is around 1.0 or 2.0
    max_boost = np.max(boost_matrix)
    if max_boost > 0:
        boost_matrix = boost_matrix / max_boost
        
    # Combine
    # Heuristic = Base + Alpha * Boost
    # If Alpha is large, the heuristic becomes dominated by the "good edges".
    # This effectively tells the ACO: "These edges are very attractive."
    
    alpha = 5.0
    heuristic = heuristic + alpha * boost_matrix
    
    # Ensure positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # 3. Validate with a quick ACO run to ensure it's not pathological
    # If the heuristic is too "binary", ACO might get stuck.
    # But the base 1/d provides some diversity.
    
    # Optional: Fine-tune alpha if time permits?
    # Let's stick with a robust alpha.
    
    # Final safety checks
    if not np.all(np.isfinite(heuristic)):
        # Fallback to simple 1/d
        heuristic = 1.0 / (distance_matrix + 1e-9)
        
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
