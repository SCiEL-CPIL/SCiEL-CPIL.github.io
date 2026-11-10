"""
Structural Heuristic for ACO CVRP using Clarke-Wright Savings.

Strategy:
1. Compute the Clarke-Wright savings for all pairs of customers.
2. Run a fast, deterministic greedy construction heuristic that respects capacity.
3. Identify the edges used in this high-quality greedy solution.
4. Construct the heuristic matrix such that these "good" edges have very high attractiveness,
   while other edges fall back to a standard inverse-distance heuristic.
5. This guides the ACO ants toward the structural skeleton of a good solution immediately,
   reducing the search space effectively.
"""

import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses a structural approach: identifies high-value edges via a greedy
    Clarke-Wright savings algorithm and boosts their heuristic values.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    n = distance_matrix.shape[0]
    
    # 1. Base Heuristic: Inverse Distance
    # This ensures all edges have some baseline attractiveness.
    # We use a slightly sharper exponent to emphasize short distances.
    base_heuristic = 1.0 / (distance_matrix + 1e-9) ** 1.2
    
    # 2. Compute Clarke-Wright Savings
    # savings[i, j] = d(i, depot) + d(depot, j) - d(i, j)
    # High savings indicate that linking i and j saves significant distance
    # compared to serving them separately from the depot.
    d_depot = distance_matrix[0, :]  # Distance from depot to all nodes
    # savings[i, j] = d_depot[i] + d_depot[j] - distance_matrix[i, j]
    savings = np.zeros((n, n))
    # Vectorized computation for i, j in 1..n-1
    # We only care about customer-customer links for savings
    for i in range(1, n):
        for j in range(i + 1, n):
            s = d_depot[i] + d_depot[j] - distance_matrix[i, j]
            savings[i, j] = s
            savings[j, i] = s
            
    # 3. Greedy Construction with Capacity Constraints
    # We simulate a simple constructive heuristic to find a "good" set of edges.
    # This acts as a surrogate for the optimal tour structure.
    
    # State: 
    # - routes: list of lists of customer indices (excluding depot)
    # - We start with each customer as a separate route: [i]
    # - We merge routes if the savings are high and capacity allows.
    
    # Initialize routes
    routes = [[i] for i in range(1, n)]
    # Track which route each customer belongs to? 
    # For n=50, a simple list of lists is fast enough.
    # We need to check capacity: sum(demands[r]) <= capacity
    
    # Precompute route demands
    route_demands = np.array([sum(demands[r]) for r in routes])
    
    # Create a list of potential merges sorted by savings (descending)
    # Only consider merges between different routes
    potential_merges = []
    for i in range(1, n):
        for j in range(i + 1, n):
            if savings[i, j] > 0:
                # Find which routes i and j belong to
                # This is O(n) per pair, O(n^3) total. n=50 -> 125k ops. Fast.
                ri = -1
                rj = -1
                for k, r in enumerate(routes):
                    if i in r:
                        ri = k
                    if j in r:
                        rj = k
                if ri != -1 and rj != -1 and ri != rj:
                    potential_merges.append((savings[i, j], ri, rj, i, j))
    
    # Sort by savings descending
    potential_merges.sort(key=lambda x: x[0], reverse=True)
    
    # Active edges mask
    active_edges = np.zeros((n, n), dtype=bool)
    
    # Greedy merge loop
    # We maintain a mapping from customer -> route index
    cust_to_route = {}
    for k, r in enumerate(routes):
        for c in r:
            cust_to_route[c] = k
            
    # Note: routes list indices change when we merge. 
    # To avoid complex index management, we'll just mark edges as we decide to merge.
    # We don't actually need to maintain the route structure perfectly if we just want the edges.
    # However, capacity check requires knowing the current route contents.
    # Let's use a Union-Find like structure or just update the lists.
    
    # Simpler approach for n=50:
    # Just iterate through sorted savings.
    # Keep a set of "current routes" as frozensets? No, lists are easier to check capacity.
    
    # Let's restart the greedy logic with a cleaner structure
    # routes is a list of lists.
    # We'll iterate through potential merges.
    # If we accept a merge, we update the routes list and route_demands.
    
    # To make "find route index" fast, we'll maintain a dict cust -> route_idx
    # But merging changes indices. 
    # Alternative: Don't maintain indices. Just check if i and j are in the same route by scanning.
    # n=50, max routes=50. Scan is O(50). 
    # Total merges <= 49.
    # Total checks = len(potential_merges) ~ 1200.
    # 1200 * 50 (scan) * 2 (check i and j) = 120k ops. Very fast.
    
    # Re-initialize
    routes = [[i] for i in range(1, n)]
    
    for s_val, ri_old, rj_old, i, j in potential_merges:
        if time.time() > deadline - 10:
            break
            
        # Check if i and j are still in different routes
        # Find current route of i and j
        route_i = None
        route_j = None
        idx_i = -1
        idx_j = -1
        
        for idx, r in enumerate(routes):
            if i in r:
                route_i = r
                idx_i = idx
            if j in r:
                route_j = r
                idx_j = idx
                
        if route_i is None or route_j is None:
            continue
        if idx_i == idx_j:
            continue # Already in same route
            
        # Check capacity
        # Demand of route_i + Demand of route_j <= capacity?
        # We need to track demands. Let's just sum them on the fly since n is small.
        demand_i = sum(demands[c] for c in route_i)
        demand_j = sum(demands[c] for c in route_j)
        
        if demand_i + demand_j <= capacity:
            # Merge
            # Mark edges i-j and j-i as active?
            # Actually, we want to boost the edges that connect the two routes.
            # The edge (i, j) is the one being added.
            active_edges[i, j] = True
            active_edges[j, i] = True
            
            # Also, we should boost the "internal" edges of the new combined route?
            # No, the heuristic is for the *next* step.
            # If we boost i->j, the ant is likely to go i->j.
            # But what about the other edges in the route?
            # The greedy solution defines a set of edges that form a good tour.
            # We should boost ALL edges in the greedy solution.
            
            # Merge the routes
            new_route = route_i + route_j
            # Remove old routes and add new one
            # To keep indices stable for the next iteration, we replace one and remove the other
            routes[idx_i] = new_route
            if idx_j > idx_i:
                routes.pop(idx_j)
            else:
                routes.pop(idx_i)
                routes.insert(idx_j, new_route)
                
            # Note: The list `routes` has changed size.
            # The next iteration will scan the updated list.
            
    # 4. Construct Final Heuristic
    # Start with base heuristic
    final_heuristic = base_heuristic.copy()
    
    # Boost active edges
    # We want the active edges to be significantly more attractive.
    # Let's multiply them by a large factor, e.g., 10x or 100x.
    # Since pheromone starts at 1 and heuristic is multiplied, a 10x boost is strong.
    # Let's try a dynamic boost: 
    # If an edge is in the greedy solution, it's "good".
    # We set heuristic[i, j] = max(base_heuristic[i, j], 10 * base_heuristic[i, j])
    # Actually, just multiplying by 10 is enough.
    
    # However, we must be careful. The ACO uses P ~ pheromone * heuristic.
    # If we boost specific edges too much, the ants might get stuck if capacity prevents them.
    # But the greedy solution respected capacity, so these edges form valid routes.
    # The ants might still fail if they visit nodes in a different order, but the heuristic
    # will guide them to try these edges first.
    
    # Let's apply a strong boost to the edges found by the greedy algorithm.
    # Factor of 5.0 is a reasonable starting point.
    boost_factor = 5.0
    final_heuristic[active_edges] *= boost_factor
    
    # Additionally, we can add a small bias for edges with high savings even if not in the greedy solution
    # due to capacity constraints, but let's stick to the structural edges first.
    
    # Ensure positivity
    final_heuristic = np.maximum(final_heuristic, 1e-9)
    
    return final_heuristic
