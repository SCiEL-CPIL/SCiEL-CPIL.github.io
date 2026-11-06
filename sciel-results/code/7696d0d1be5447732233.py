"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0  # safety margin of 2s before 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing using Clarke-Wright Savings.
    
    Strategy:
    1. If time is low or no feasible nodes, fallback to nearest feasible or depot.
    2. Compute savings for all pairs of unvisited nodes and depot.
    3. Greedily merge routes based on highest savings, respecting capacity constraints.
    4. Identify the route containing 'current_node' (which starts at depot).
    5. Return the next node in that route sequence.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return int(depot)
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if not np.any(feasible_mask):
            return int(depot)
        feasible_nodes = unvisited_nodes[feasible_mask]
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Filter to feasible nodes
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    n = len(feasible_nodes)
    
    # If only one node, it must be that one
    if n == 1:
        return int(feasible_nodes[0])
    
    # Initialize routes: each node is its own route
    # Route structure: list of nodes in order
    routes = [[int(node)] for node in feasible_nodes]
    
    # Calculate capacity used for each route
    route_caps = [int(demands[node]) for node in feasible_nodes]
    
    # Calculate savings
    # Savings(i, j) = d(depot, i) + d(depot, j) - d(i, j)
    # We only consider pairs that are in different routes and can be merged
    # We need to be careful about the direction of merging.
    # Standard Clarke-Wright merges i and j if they are at the end of their respective routes.
    # However, since we start with single nodes, any pair is valid.
    # As routes grow, we can only merge if the connection is between the ends.
    
    # To keep it simple and fast for 50 nodes:
    # 1. Calculate all savings for pairs of nodes.
    # 2. Sort savings in descending order.
    # 3. Iterate and try to merge.
    
    # Precompute savings matrix for feasible nodes
    # Note: distance_matrix is global, we index by node ID
    savings = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            node_i = feasible_nodes[i]
            node_j = feasible_nodes[j]
            s = distance_matrix[depot, node_i] + distance_matrix[depot, node_j] - distance_matrix[node_i, node_j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Get indices of savings sorted by value
    # We want to consider all pairs (i, j) where i != j
    # Flatten and sort
    flat_savings = savings.flatten()
    sorted_indices = np.argsort(flat_savings)[::-1]
    
    # Track which route each node belongs to
    # route_id_to_indices: dict mapping route_id -> list of node indices in feasible_nodes
    # Actually, let's track routes as lists of node IDs
    # And a mapping from node ID to route index
    node_to_route = {int(node): idx for idx, node in enumerate(feasible_nodes)}
    
    # routes is a list of lists of node IDs
    # route_caps is a list of capacities
    
    for idx in sorted_indices:
        if _time_exceeded():
            break
            
        i = idx // n
        j = idx % n
        
        if i == j:
            continue
            
        # Get node IDs
        node_i = feasible_nodes[i]
        node_j = feasible_nodes[j]
        
        # Get current routes
        route_i_idx = node_to_route.get(node_i)
        route_j_idx = node_to_route.get(node_j)
        
        if route_i_idx is None or route_j_idx is None:
            continue
            
        if route_i_idx == route_j_idx:
            continue
            
        # Check capacity
        cap_i = route_caps[route_i_idx]
        cap_j = route_caps[route_j_idx]
        if cap_i + cap_j > rest_capacity:
            continue
            
        # Check if nodes are at the ends of their respective routes
        # This is a simplified check. In full CW, we ensure we merge tails.
        # Since we start with single nodes, they are always at ends.
        # As we merge, we need to ensure we don't create a cycle and we merge ends.
        # Let's assume we can always merge if we pick the right end.
        # To be safe, we only merge if node_i is at an end of route_i and node_j is at an end of route_j.
        
        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]
        
        # Find position of node_i in route_i
        pos_i = -1
        for k, node in enumerate(route_i):
            if node == node_i:
                pos_i = k
                break
                
        pos_j = -1
        for k, node in enumerate(route_j):
            if node == node_j:
                pos_j = k
                break
        
        if pos_i == -1 or pos_j == -1:
            continue
            
        # Check if they are at the ends
        is_end_i = (pos_i == 0 or pos_i == len(route_i) - 1)
        is_end_j = (pos_j == 0 or pos_j == len(route_j) - 1)
        
        if not (is_end_i and is_end_j):
            continue
            
        # Determine merge order
        # We want to connect node_i and node_j.
        # If node_i is at the end (last element) of route_i, we append route_j to route_i (or reverse route_j if node_j is at start)
        # If node_i is at the start (first element) of route_i, we prepend route_j to route_i (or reverse route_j if node_j is at end)
        
        # Let's standardize: we merge route_j into route_i
        # Case 1: node_i is last in route_i
        if pos_i == len(route_i) - 1:
            # route_i ends with node_i
            if pos_j == 0:
                # route_j starts with node_j -> append route_j to route_i
                new_route = route_i + route_j
            else:
                # route_j ends with node_j -> append reverse(route_j) to route_i
                new_route = route_i + route_j[::-1]
        else:
            # node_i is first in route_i
            if pos_j == len(route_j) - 1:
                # route_j ends with node_j -> prepend route_j to route_i
                new_route = route_j + route_i
            else:
                # route_j starts with node_j -> prepend reverse(route_j) to route_i
                new_route = route_j[::-1] + route_i
        
        # Update routes
        routes[route_i_idx] = new_route
        route_caps[route_i_idx] = cap_i + cap_j
        
        # Remove route_j
        # We need to update node_to_route for all nodes in route_j
        for node in route_j:
            node_to_route[node] = route_i_idx
            
        # Mark route_j as empty/removed
        routes[route_j_idx] = []
        route_caps[route_j_idx] = 0
        
    # Now, find the route containing current_node
    # Note: current_node might be depot or a visited node? 
    # The problem says "unvisited_nodes". current_node is the one we are at.
    # If current_node is depot, it's not in feasible_nodes.
    # If current_node is a customer, it IS in feasible_nodes (since it's unvisited? No, current_node is usually the last visited or depot).
    # Wait, the function signature implies we are AT current_node and need to pick NEXT.
    # The "unvisited_nodes" are the ones not yet visited.
    # So current_node is NOT in unvisited_nodes.
    # Therefore, current_node is NOT in feasible_nodes.
    # My previous logic "Identify the route containing current_node" is flawed because current_node is not in the set we are routing.
    
    # Correction:
    # The heuristic constructs routes for the UNVISITED nodes.
    # The vehicle is currently at `current_node`.
    # We need to pick the next node.
    # In a standard CVRP construction (like Sweep or Nearest Neighbor), we build the route from the depot.
    # Here, we are in the middle of a route? Or starting a new one?
    # "rest_capacity" implies we have capacity left.
    # If we return to depot, we close the route.
    
    # If we use Clarke-Wright on unvisited nodes, we get a set of routes starting from depot.
    # But we are not at the depot. We are at current_node.
    # This makes direct application of CW tricky because the "start" is fixed at depot for the tour, but we are at current_node.
    
    # Alternative Interpretation:
    # The evaluator likely runs a loop:
    # node = depot
    # while unvisited:
    #    next_node = select_next_node(...)
    #    if next_node == depot: break
    #    visit next_node
    #    update state
    
    # So, if current_node is depot, we are starting a route.
    # If current_node is a customer, we are continuing a route.
    
    # If we use CW on the unvisited nodes, we generate a set of complete routes (cycles starting/ending at depot).
    # We need to pick the next node for the CURRENT route.
    
    # Strategy Adjustment:
    # 1. Generate CW routes for unvisited nodes.
    # 2. If current_node is depot:
    #    - Find the route with the lowest total length? Or just the first one?
    #    - Actually, if we are at depot, we should pick the first node of the "best" route.
    #    - But which route is "best"? They are all valid.
    #    - Maybe we should just pick the node that is part of the route that would be served first?
    #    - Simpler: If at depot, pick the node that is the start of the route with the highest savings potential?
    #    - Or just pick the nearest feasible node if at depot?
    
    # 3. If current_node is NOT depot:
    #    - We are in a route.
    #    - The CW algorithm generates routes starting from depot.
    #    - This doesn't directly tell us what to do next from current_node.
    
    # Let's rethink. The "Savings" inspiration suggests using savings to guide the choice.
    # Maybe we can compute savings for the pair (current_node, candidate) vs (current_node, depot) + (depot, candidate)?
    # No, that's not right.
    
    # Let's stick to a robust heuristic that works for both cases:
    # 1. If at depot: Use Nearest Neighbor with Savings lookahead? Or just Nearest Neighbor?
    # 2. If not at depot: Use a score that favors nodes that allow closing the route soon if capacity is low, or nodes that are close.
    
    # Given the constraint "STRUCTURAL change", and the inspiration "Savings-based construction",
    # perhaps the intended approach is:
    # Treat the current state as "We have a partial route ending at current_node".
    # We want to extend this route.
    # We can use the savings idea to decide if we should continue or return to depot.
    # Savings of adding node j to current route ending at i:
    # Cost(i->j->depot) vs Cost(i->depot) + Cost(depot->j->depot)?
    # No, that's not quite it.
    
    # Let's try a different structural change: **Greedy with 2-Opt Local Search on the current route segment**.
    # But the function only returns ONE node.
    
    # Let's go back to the **Clarke-Wright** idea but adapt it:
    # We compute the savings for all pairs of unvisited nodes.
    # We also consider the "current node" as a virtual node in the savings calculation?
    
    # Actually, a simpler and very effective heuristic for CVRP is **Sweep**.
    # But that requires angular sorting.
    
    # Let's try **Nearest Neighbor with Capacity Penalty**.
    # Score = distance(current, next) + penalty if demand(next) is large relative to rest_capacity.
    # This is similar to the current program but maybe the current one is too complex or the weights are bad.
    
    # The directive says: "Model the routing decision as a shortest path problem... Dijkstra's".
    # This is a strong hint.
    
    # Let's implement a **Dijkstra-based lookahead** for the next 1-2 steps.
    # State: (node, capacity_used)
    # But capacity_used is not passed, only rest_capacity.
    # We can assume we want to minimize distance to visit K nodes and return to depot.
    
    # Given the 150s limit and 50 nodes, a full Dijkstra on (node, cap) is feasible if we limit the depth.
    # Depth 2:
    # For each candidate node j:
    #   cost_j = dist(current, j)
    #   For each node k != j:
    #     if demand(j) + demand(k) <= rest_capacity:
    #       cost_jk = dist(current, j) + dist(j, k) + dist(k, depot)
    #   Take min cost_jk over k, or cost_j + dist(j, depot) if no k.
    #   Score j by this min cost.
    
    # This is a 2-step lookahead.
    
    # Let's implement this 2-step lookahead. It is structurally different from the 1-step scoring of the current program.
    
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
        
    feasible_nodes = unvisited_nodes[feasible_mask]
    n = len(feasible_nodes)
    
    if n == 1:
        return int(feasible_nodes[0])
    
    # 2-step lookahead
    # For each node i in feasible_nodes:
    #   best_cost_i = dist(current, i) + dist(i, depot)  # If we stop after i
    #   For each node j in feasible_nodes, j != i:
    #     if demands[i] + demands[j] <= rest_capacity:
    #       cost_ij = dist(current, i) + dist(i, j) + dist(j, depot)
    #       best_cost_i = min(best_cost_i, cost_ij)
    #   Score[i] = best_cost_i
    # Return argmin Score
    
    scores = np.zeros(n)
    
    for i_idx, node_i in enumerate(feasible_nodes):
        d_ci = distance_matrix[current_node, node_i]
        d_id = distance_matrix[node_i, depot]
        
        # Option 1: Go to i, then depot
        best_cost = d_ci + d_id
        
        # Option 2: Go to i, then j, then depot
        # Only consider if capacity allows
        if demands[node_i] < rest_capacity:
            remaining_cap_after_i = rest_capacity - demands[node_i]
            # Find j that minimizes dist(i, j) + dist(j, depot) subject to demand(j) <= remaining_cap_after_i
            # We can vectorize this
            mask_j = demands[feasible_nodes] <= remaining_cap_after_i
            # Exclude i
            mask_j[i_idx] = False
            
            if np.any(mask_j):
                valid_js = feasible_nodes[mask_j]
                d_ij = distance_matrix[node_i, valid_js]
                d_jd = distance_matrix[valid_js, depot]
                total_costs = d_ij + d_jd
                min_cost_ij = np.min(total_costs)
                cost_ij = d_ci + min_cost_ij
                if cost_ij < best_cost:
                    best_cost = cost_ij
        
        scores[i_idx] = best_cost
        
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
