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

def _lagrangian_heuristic(distance_matrix, demands, capacity, coordinates, start_time, deadline):
    n = distance_matrix.shape[0]
    
    # 1. Solve Lagrangian Relaxation
    # Objective: Min sum d_ij x_ij + sum lambda * max(0, load_k - capacity)
    # Approximation: Use subgradient method on a simplified model or just tune lambda
    # to balance distance and number of routes.
    
    # Initial lambda
    lambda_ = 1.0
    
    # We will simulate a "relaxed" tour construction for different lambdas
    # and pick the one that yields the best ACO performance.
    
    candidates = []
    
    # Try a range of lambda values
    # lambda represents the cost per unit of capacity violation or penalty for using a new route.
    # In a simple heuristic, we can use this to weight the "start new route" cost.
    # Or, we can use it to penalize edges that lead to high load?
    # Let's stick to the directive: "solves the relaxed CVRP... to identify edges".
    
    # A common approach:
    # Solve TSP on the full set (ignoring capacity).
    # This gives a set of edges.
    # If we break this TSP tour into routes respecting capacity, we get a CVRP solution.
    # The "Lagrangian" part could be adjusting the weights to favor shorter tours that can be easily split.
    
    # Let's implement a multi-start nearest neighbor + 2-opt for the "relaxed" problem (TSP)
    # but with a penalty for long edges that might cause capacity issues?
    # Actually, the directive says: "penalizing load violations via dual variables".
    
    # Let's approximate the dual update:
    # Start with lambda = 0.
    # Construct a solution.
    # If capacity is violated (which it will be if we treat it as TSP), increase lambda.
    # But we want a *heuristic matrix* for ACO.
    
    # Strategy:
    # 1. Generate a TSP tour (ignoring capacity).
    # 2. Split the tour into routes by capacity.
    # 3. The edges in the TSP tour that are preserved in the final routes are "good".
    # 4. Edges that were cut are "bad".
    # 5. We can vary the "start point" or the "splitting strategy" to get different edge sets.
    # 6. We can also use a "penalized" distance matrix: d'_ij = d_ij + lambda * (some_load_term).
    
    # Let's try varying lambda in a "penalized distance" matrix and running a simple TSP heuristic.
    # The "load term" is hard to define for a single edge.
    # Alternative: Use the savings heuristic but with a modified savings value that includes a penalty.
    # Savings(i,j) = d(0,i) + d(0,j) - d(i,j) - lambda * (demand_i + demand_j - capacity)
    # If demand_i + demand_j > capacity, the penalty is positive, reducing the savings.
    
    best_score = float('inf')
    best_heuristic = None
    
    # Range of lambdas to test
    # Since demands are 1-9 and capacity 50, violations are small.
    # Let's test lambdas: 0, 0.1, 0.5, 1.0, 5.0
    
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    for lam in lambdas:
        if time.time() > deadline - 10: # Leave buffer
            break
            
        # Construct heuristic based on penalized savings
        # We need to run the savings algorithm with the penalty.
        
        # For speed, we can just compute the "attractiveness" directly from the penalized savings logic
        # without running the full greedy merge, or run a fast version.
        
        # Let's run a fast savings-based construction to get edges.
        edges = _get_edges_penalized_savings(distance_matrix, demands, capacity, lam)
        
        # Construct heuristic matrix
        h = np.full((n, n), 0.01, dtype=float)
        for i, j in edges:
            h[i, j] = 1.0
            # Also add reverse? In savings, direction matters for the specific merge, 
            # but for ACO, if i->j is good, j->i might be good in reverse route.
            # Let's boost both to be safe, or just the directed ones.
            # The ACO moves i->j. If the route is 0->...->i->j->...->0, then i->j is used.
            # If the route is reversed, j->i is used.
            # Since we don't know the direction in the final ACO run, boosting both is safer.
            h[j, i] = 1.0
            
        h = np.where(h > 0, h, 1e-9)
        
        # Evaluate with ACO
        try:
            score = _run_aco(distance_matrix, coordinates, demands, capacity, h, seed=42)
            if score < best_score:
                best_score = score
                best_heuristic = h.copy()
        except Exception:
            continue
            
    if best_heuristic is None:
        # Fallback
        best_heuristic = 1.0 / (distance_matrix + 1e-9)
        
    return best_heuristic

def _get_edges_penalized_savings(distance_matrix, demands, capacity, lam):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    pred = [-1] * n
    succ = [-1] * n
    
    for i in customers:
        pred[i] = 0
        succ[i] = 0
        
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                # Penalty term
                # If merging i and j exceeds capacity, penalty.
                # But in savings, we merge routes. The demand of the merged route is sum of demands.
                # We don't know the full route demand here easily without tracking.
                # Let's just use a simple heuristic: if demands[i] + demands[j] > capacity, penalize.
                # This is a local approximation.
                penalty = 0.0
                if demands[i] + demands[j] > capacity:
                    penalty = lam * (demands[i] + demands[j] - capacity)
                
                s -= penalty
                savings.append((s, i, j))
                
    savings.sort(reverse=True)
    
    route_id = list(range(n))
    route_demand = [0.0] * n
    for i in customers:
        route_demand[i] = demands[i]
        
    edges = set()
    
    for s, i, j in savings:
        if s <= 0:
            break
            
        if succ[i] != 0:
            continue
        if pred[j] != 0:
            continue
        if route_id[i] == route_id[j]:
            continue
            
        new_demand = route_demand[i] + route_demand[j]
        if new_demand > capacity:
            continue
            
        succ[i] = j
        pred[j] = i
        
        r_j = route_id[j]
        r_i = route_id[i]
        for k in range(1, n):
            if route_id[k] == r_j:
                route_id[k] = r_i
                
        route_demand[r_i] = new_demand
        route_demand[r_j] = 0
        
        edges.add((i, j))
        
    final_edges = set()
    for i in customers:
        p = pred[i]
        if p != -1:
            final_edges.add((p, i))
        s = succ[i]
        if s != -1:
            final_edges.add((i, s))
            
    return final_edges

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280
    
    n = distance_matrix.shape[0]
    
    # Use Lagrangian-inspired heuristic
    best_heuristic = _lagrangian_heuristic(distance_matrix, demands, capacity, coordinates, start_time, deadline)
    
    # Safety checks
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = 1.0 / (distance_matrix + 1e-9)
        
    best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
