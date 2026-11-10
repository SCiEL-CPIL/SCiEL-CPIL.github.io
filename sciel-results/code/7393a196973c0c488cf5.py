"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precomputation cache for savings-based heuristic
# Key: tuple of (current_node, rest_capacity, frozenset(unvisited_nodes)) -> next_node
# But since unvisited_nodes changes, we use a simpler per-instance approach.

# We'll implement a savings-based heuristic with 1-step lookahead.
# The idea: for each feasible candidate, compute a score based on:
#   1. Immediate distance from current node to candidate
#   2. The "savings" of visiting candidate vs. going back to depot and then to candidate
#   3. A lookahead term: best distance from candidate to any other feasible node (or depot)

# Since the function is called repeatedly during route construction, we need it to be fast.
# We'll precompute some things per call but keep it lightweight.

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit using a savings-based heuristic with lookahead.
    Returns depot (0) to close the route if no feasible node or if closing is better.
    """
    # Convert to numpy arrays for speed
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    n_unvisited = len(unvisited)
    
    if n_unvisited == 0:
        return depot
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get demands of unvisited nodes
    demands_unvisited = demands[unvisited]
    
    # Feasibility: demand must fit in remaining capacity
    feasible_mask = demands_unvisited <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return depot
    
    feasible_unvisited = unvisited[feasible_indices]
    dist_feasible = dist_from_current[feasible_indices]
    demands_feasible = demands_unvisited[feasible_indices]
    
    # If only one feasible node, go there
    if len(feasible_unvisited) == 1:
        return int(feasible_unvisited[0])
    
    # Compute savings: for each feasible candidate i, the saving is
    #   d(current, depot) + d(depot, i) - d(current, i)
    # But we want to minimize total route length, so we prefer nodes with high savings
    # (i.e., where going to i now saves more than going back to depot first).
    # Actually, the classic savings is d(depot, i) + d(depot, j) - d(i, j) for merging routes.
    # Here, we're building a route, so we want to pick the next node that minimizes
    # the total route length. A good heuristic is to consider:
    #   cost(i) = d(current, i) + d(i, best_next_after_i)
    # where best_next_after_i is the best (closest feasible) node after i, or depot.
    
    # For speed, we'll use a 1-step lookahead:
    # For each candidate i, estimate the cost of going current -> i -> (best next or depot)
    # The "best next" for i is the closest feasible node j != i, or depot if no feasible j.
    
    # Precompute: for each candidate, the minimum distance to any other feasible candidate or depot
    # This is O(n^2) but n is at most 50, so it's fine.
    
    n_feas = len(feasible_unvisited)
    
    # Matrix of distances between feasible candidates
    dist_matrix_feasible = distance_matrix[np.ix_(feasible_unvisited, feasible_unvisited)]
    
    # For each candidate i, the best next step cost (excluding itself)
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(dist_matrix_feasible, np.inf)
    
    # Also consider going to depot as an option
    dist_to_depot = distance_matrix[feasible_unvisited, depot]
    
    # The best next cost for candidate i is min(min over j!=i of d(i,j), d(i, depot))
    min_dist_to_other = np.min(dist_matrix_feasible, axis=1)
    best_next_cost = np.minimum(min_dist_to_other, dist_to_depot)
    
    # Total estimated cost if we pick candidate i:
    #   d(current, i) + best_next_cost[i]
    # We want to minimize this.
    
    # But we also want to consider capacity: after visiting i, the remaining capacity
    # is rest_capacity - demand[i]. We should prefer nodes that leave more flexibility.
    # A simple way: add a small penalty for high demand.
    
    # Let's use a weighted score:
    #   score(i) = d(current, i) + best_next_cost[i] + alpha * demand[i]
    # where alpha is a small weight to prefer lower-demand nodes when costs are close.
    
    alpha = 0.1  # Small weight for demand
    
    scores = dist_feasible + best_next_cost + alpha * demands_feasible
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_unvisited[best_idx])

# EVOLVE-BLOCK-END
