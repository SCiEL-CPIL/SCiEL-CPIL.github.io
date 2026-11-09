"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _solve_cvrp_heuristic(distance_matrix, coordinates, demands, capacity):
    """
    Solves CVRP using a fast heuristic (Clarke-Wright Savings) to find a good set of edges.
    Returns a set of (i, j) tuples that form the routes.
    """
    n = distance_matrix.shape[0]
    # Node 0 is depot
    customers = list(range(1, n))
    
    # Initialize routes: each customer is its own route
    # route[i] is a list of nodes in the route, e.g., [0, i, 0] is not quite right for savings.
    # Let's use a standard savings approach.
    # Predecessor and successor arrays
    pred = [-1] * n
    succ = [-1] * n
    
    # Initialize: each customer i has route 0 -> i -> 0
    for i in customers:
        pred[i] = 0
        succ[i] = 0
        
    # Calculate savings for merging route ending at i and starting at j
    # s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings.append((s, i, j))
                
    # Sort savings in descending order
    savings.sort(reverse=True)
    
    # Track current routes to check capacity
    # route_nodes[r] = list of nodes in route r
    # We need a way to find which route a node belongs to and its position
    # Simpler: maintain a list of routes as lists of nodes.
    # routes = {i: [0, i, 0] for i in customers}
    # But merging is easier if we keep pred/succ and a route_id.
    
    route_id = list(range(n)) # route_id[i] is the id of the route containing i
    # Initially, each customer i is in route i. Depot is in route 0? No, depot is shared.
    # Let's say route_id[i] = i for i in 1..n-1.
    
    # To check capacity, we need the total demand of a route.
    route_demand = [0.0] * n
    for i in customers:
        route_demand[i] = demands[i]
        
    # To check if i is the end of its route and j is the start of its route
    # End of route: succ[i] == 0
    # Start of route: pred[j] == 0
    
    # We also need to ensure we don't create a cycle before including depot (shouldn't happen with this method)
    
    edges = set()
    
    for s, i, j in savings:
        if s <= 0:
            break
            
        # Check if i is the end of its route
        if succ[i] != 0:
            continue
        # Check if j is the start of its route
        if pred[j] != 0:
            continue
            
        # Check if they are in the same route (cycle)
        if route_id[i] == route_id[j]:
            continue
            
        # Check capacity
        new_demand = route_demand[i] + route_demand[j]
        if new_demand > capacity:
            continue
            
        # Merge
        # i is end of route i, j is start of route j
        # Connect i -> j
        succ[i] = j
        pred[j] = i
        
        # Update route_id for all nodes in route j to be route_id[i]
        # This is O(n) per merge, total O(n^2) which is fine for n=50
        r_j = route_id[j]
        r_i = route_id[i]
        for k in range(1, n):
            if route_id[k] == r_j:
                route_id[k] = r_i
                
        route_demand[r_i] = new_demand
        route_demand[r_j] = 0 # Invalidate old route id
        
        edges.add((i, j))
        
    # Extract edges from final pred/succ structure
    final_edges = set()
    for i in customers:
        # Edge from pred[i] to i
        p = pred[i]
        if p != -1:
            final_edges.add((p, i))
        # Edge from i to succ[i]
        s = succ[i]
        if s != -1:
            final_edges.add((i, s))
            
    return final_edges

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

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s buffer for safety
    
    n = distance_matrix.shape[0]
    
    # 1. Solve CVRP heuristic to get "good" edges
    good_edges = _solve_cvrp_heuristic(distance_matrix, coordinates, demands, capacity)
    
    # 2. Construct heuristic matrix
    # Base: very low value (repel)
    # Good edges: very high value (attract)
    # We want to guide the ACO to these edges.
    # If we set good edges to 1.0 and others to 0.001, the ACO will strongly prefer them.
    # But we should also consider distance slightly to break ties or handle variations?
    # The task says "Heuristic entries that are not positive become 1e-9".
    
    heuristic = np.full((n, n), 0.01, dtype=float)
    
    # Boost good edges
    for i, j in good_edges:
        heuristic[i, j] = 1.0
        # Also consider reverse? No, CVRP is directed in routing, but distance matrix is symmetric.
        # The ACO moves from i to j. If (i,j) is in the solution, boost h[i,j].
        # Since distance matrix is symmetric, and routes are undirected in concept but directed in traversal,
        # and we found edges (u,v) meaning u->v or v->u? 
        # In savings, we connect i->j. So h[i,j] is boosted.
        # The reverse edge j->i might also be part of a valid route in a different orientation, 
        # but typically in TSP/CVRP with symmetric dist, if i->j is good, j->i is good for the reverse tour.
        # However, the specific route direction matters for capacity.
        # Let's just boost the specific directed edges found.
        
    # Also, it might be beneficial to have a slight distance bias for non-good edges
    # so that if the ACO is forced off the "good" path, it picks the nearest neighbor.
    dm_safe = distance_matrix + 1e-9
    dist_bias = 1.0 / dm_safe
    dist_bias = dist_bias / np.max(dist_bias) # Normalize to [0,1]
    
    # Combine: 
    # If edge is good: 1.0
    # If edge is not good: small base + distance bias
    # This ensures that if the "good" structure is broken, the ant still makes a reasonable move.
    
    mask_good = np.zeros((n, n), dtype=bool)
    for i, j in good_edges:
        mask_good[i, j] = True
        
    heuristic = np.where(mask_good, 1.0, 0.01 + 0.01 * dist_bias)
    
    # Ensure positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Optional: Verify with a quick ACO run?
    # The current program spends a lot of time grid searching.
    # This approach is O(n^2 log n) for savings + O(n^2) for matrix creation.
    # It is extremely fast. We have plenty of time.
    # We could try a few variations of the "boost" strength or base value.
    
    best_score = float('inf')
    best_heuristic = heuristic
    
    # Try a few variations of the heuristic strength
    # Variation 1: Strong attractor (1.0 vs 0.01)
    # Variation 2: Medium attractor (1.0 vs 0.1)
    # Variation 3: Pure good edges (1.0 vs 1e-9)
    
    variations = [
        (1.0, 0.01, 0.01), # Good, Base, DistWeight
        (1.0, 0.1, 0.0),   # Good, Base, No Dist
        (10.0, 0.1, 0.0),  # Super Good, Base
        (1.0, 0.001, 0.0), # Good, Tiny Base
    ]
    
    for g_val, base_val, dw in variations:
        if time.time() > deadline:
            break
            
        h_var = np.where(mask_good, g_val, base_val)
        if dw > 0:
            h_var = h_var + dw * dist_bias
            # Only add dist bias to non-good edges? Or all?
            # If added to good edges, it might dilute the signal.
            # Let's only add to non-good.
            h_var = np.where(mask_good, g_val, base_val + dw * dist_bias)
            
        h_var = np.where(h_var > 0, h_var, 1e-9)
        
        try:
            score = _run_aco(distance_matrix, coordinates, demands, capacity, h_var, seed=42)
            if score < best_score:
                best_score = score
                best_heuristic = h_var
        except Exception:
            continue
            
    # Final safety checks
    if not np.all(np.isfinite(best_heuristic)):
        # Fallback to simple 1/d
        best_heuristic = 1.0 / (distance_matrix + 1e-9)
        
    best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
