# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    The heuristic is designed to guide ants towards nodes that offer high prize
    per unit distance, while also considering the feasibility constraint (max_len).
    
    Strategy:
    - For each edge (i, j), estimate the 'value' of going from i to j.
    - The value should reflect:
      * The prize at node j (higher is better).
      * The cost of the edge d[i,j] (lower is better).
      * The feasibility: can we go i -> j -> 0 within max_len? 
        If d[i,j] + d[j,0] <= max_len, then j is reachable from i and can return.
        This is a key constraint in orienteering with a depot.
    - We combine these into a single attractiveness score.
    
    A robust approach:
    1. Compute a feasibility mask: feasible[i,j] = (d[i,j] + d[j,0] <= max_len)
    2. For feasible edges, compute attractiveness based on prize/distance ratio.
    3. For infeasible edges, set to a very small value (1e-9) so they are avoided.
    4. Ensure all values are positive and finite.
    
    Additionally, we can incorporate a global normalization to keep values in a
    reasonable range.
    
    Let's define:
    - prize_gain[i,j] = prizes[j] (the prize we get by going to j)
    - cost[i,j] = distance_matrix[i,j]
    - return_cost[j] = distance_matrix[j, 0]
    - feasible[i,j] = (distance_matrix[i,j] + distance_matrix[j, 0] <= max_len)
    
    Attractiveness = (prize_gain + epsilon) / (cost + return_cost + epsilon) for feasible edges
    This rewards high prizes and low total cost (go + return).
    
    We also add a small bonus for nodes that are closer to the start (node 0) initially,
    but since the ant starts at 0, the first move's cost is just d[0,j]. The heuristic
    is static, so we can't easily incorporate path length so far. However, the ACO
    algorithm will handle the path length constraint dynamically.
    
    To make the heuristic more informative, we can also consider the 'opportunity cost':
    if a node is very far, it might consume too much of the max_len budget. The
    feasibility check handles this partially.
    
    Let's also normalize the prizes and distances to avoid scale issues.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    
    # Ensure prize[0] is 0 (as per problem statement, but let's be safe)
    # The problem says prize 0 at node 0, so we don't need to do anything special.
    
    # Compute feasibility: can we go i -> j -> 0 within max_len?
    # Note: The problem says "length so far plus d[i,j] plus d[j,0] stays within max_len"
    # Since the heuristic is computed once, we don't know "length so far". 
    # But a necessary condition for j to ever be visited from i is that d[i,j] + d[j,0] <= max_len
    # (assuming we start at 0 and go to i first, the minimum length so far is d[0,i], but 
    #  the constraint is global). Actually, the constraint is:
    #   current_path_length + d[i,j] + d[j,0] <= max_len
    # Since current_path_length >= 0, a necessary condition is d[i,j] + d[j,0] <= max_len.
    # This is not sufficient, but it's a good filter for the heuristic.
    
    d = distance_matrix
    # d[i,j] + d[j,0] for all i,j
    # d[j,0] is the return cost from j to depot
    return_cost = d[:, 0]  # shape (n,)
    # feasible[i,j] = d[i,j] + return_cost[j] <= max_len
    feasible = (d + return_cost[None, :] <= max_len)
    
    # Compute base attractiveness
    epsilon = 1e-9
    
    # Prize at destination node j
    prize_dest = prizes[None, :]  # shape (1, n), broadcast to (n, n)
    
    # Cost of the edge
    edge_cost = d
    
    # Total cost consideration: edge cost + return cost
    total_cost = d + return_cost[None, :]
    
    # Attractiveness: prize / total_cost, but only for feasible edges
    # For infeasible edges, we want very low attractiveness
    base_attr = np.zeros_like(d)
    feasible_mask = feasible
    
    # Where feasible, compute prize/total_cost
    # Avoid division by zero
    safe_total_cost = np.where(total_cost > epsilon, total_cost, 1.0)
    attr_values = (prize_dest + epsilon) / safe_total_cost
    
    # Apply feasibility mask
    base_attr = np.where(feasible_mask, attr_values, epsilon)
    
    # Ensure all values are positive
    base_attr = np.maximum(base_attr, epsilon)
    
    # Optional: normalize to keep values in a reasonable range
    # Find max and min
    max_attr = np.max(base_attr)
    min_attr = np.min(base_attr)
    
    if max_attr > min_attr:
        # Normalize to [epsilon, 1.0]
        base_attr = epsilon + (base_attr - min_attr) / (max_attr - min_attr) * (1.0 - epsilon)
    else:
        base_attr = np.ones_like(base_attr)
    
    # Ensure no NaN or Inf
    base_attr = np.where(np.isfinite(base_attr), base_attr, epsilon)
    
    return base_attr
# EVOLVE-BLOCK-END
