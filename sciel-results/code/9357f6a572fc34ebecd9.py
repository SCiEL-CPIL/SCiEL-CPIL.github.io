"""Heuristic for capacitated vehicle routing: pick next node or close route."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a lightweight, fast heuristic that balances:
# 1. Nearest neighbor (short immediate distance)
# 2. Savings / cluster cohesion (prefer nodes close to each other, far from depot)
# 3. Capacity feasibility
# 4. Return-to-depot cost awareness

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit or return to depot (0).
    
    Parameters:
    - current_node: current node index
    - depot: depot index (0)
    - unvisited_nodes: array/list of unvisited customer indices
    - rest_capacity: remaining vehicle capacity
    - demands: array of demand for each node
    - distance_matrix: 2D array of distances
    
    Returns:
    - int: next node to visit, or 0 (depot) to close the route
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy array for fast indexing if needed
    unvisited = np.asarray(unvisited_nodes)
    
    # Filter by capacity feasibility
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        # No feasible customer; return to depot
        return int(depot)
    
    feasible_nodes = unvisited[feasible_mask]
    
    # If only one feasible node, take it (unless returning to depot is clearly better)
    if len(feasible_nodes) == 1:
        # Compare cost of going to that node and then back to depot
        # vs. returning to depot now
        # Actually, if we must visit all nodes, we should take the only feasible one
        # unless the route is already "closed" in some sense
        # For simplicity, take it if it's not too costly
        return int(feasible_nodes[0])
    
    # Score each feasible candidate
    # We want to minimize total route length. A good heuristic:
    # score = dist(current, candidate) + alpha * dist(candidate, depot) - beta * savings
    # where savings = dist(current, depot) + dist(depot, candidate) - dist(current, candidate)
    # But a simpler, effective approach: prefer nodes that are close to current
    # and also close to other unvisited nodes (cluster cohesion).
    
    # Compute distances from current to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Compute distances from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Compute "savings" of visiting this node now vs. doing it later
    # Savings = dist(current, depot) + dist(depot, candidate) - dist(current, candidate)
    # Higher savings means it's more beneficial to visit this node now
    dist_current_depot = distance_matrix[current_node, depot]
    savings = dist_current_depot + dist_to_depot - dist_from_current
    
    # Estimate cluster cohesion: for each candidate, how close is it to other unvisited nodes?
    # Use the minimum distance to any other unvisited node (excluding itself)
    # This encourages grouping nearby customers together
    n_feasible = len(feasible_nodes)
    
    # For efficiency, compute min distance to other unvisited nodes for each candidate
    # dist_to_others[i] = min distance from feasible_nodes[i] to any other unvisited node
    # We can compute this using the distance matrix
    # For each candidate i, look at distances from feasible_nodes[i] to all unvisited_nodes
    # and take the min, excluding itself
    
    # To keep it fast, we'll use a simple approximation:
    # cohesion[i] = min distance from candidate i to any other unvisited node
    # We compute this vectorized
    
    # Get the submatrix of distances from feasible_nodes to all unvisited_nodes
    # dist_feasible_to_unvisited[i, j] = distance from feasible_nodes[i] to unvisited[j]
    dist_feasible_to_unvisited = distance_matrix[np.ix_(feasible_nodes, unvisited)]
    
    # For each row i, set the diagonal (self) to inf, then take min
    # The self index in the unvisited array for feasible_nodes[i] is:
    # We need to find the position of feasible_nodes[i] in unvisited
    self_indices = np.searchsorted(unvisited, feasible_nodes)
    # But searchsorted gives the insertion point, which is the index if the value exists
    # Since feasible_nodes are a subset of unvisited, this should work
    
    # Set self distances to inf
    rows = np.arange(n_feasible)
    dist_feasible_to_unvisited[rows, self_indices] = np.inf
    
    # Min distance to any other unvisited node
    min_dist_to_other = np.min(dist_feasible_to_unvisited, axis=1)
    
    # Score: we want to minimize a combination of:
    # - dist_from_current (immediate cost)
    # - dist_to_depot (cost of returning to depot after this node, if it's the last)
    # - -savings (higher savings is better, so subtract)
    # - min_dist_to_other (closer to cluster is better, so minimize)
    
    # Normalize components for a reasonable weighting
    # Use a simple weighted sum
    w1 = 1.0   # weight for immediate distance
    w2 = 0.3   # weight for distance to depot (penalize if far)
    w3 = 0.5   # weight for savings (reward)
    w4 = 0.4   # weight for cluster cohesion
    
    # Score = w1 * dist_from_current + w2 * dist_to_depot - w3 * savings + w4 * min_dist_to_other
    # Lower score is better
    score = (w1 * dist_from_current + 
             w2 * dist_to_depot - 
             w3 * savings + 
             w4 * min_dist_to_other)
    
    # Find the best candidate
    best_idx = np.argmin(score)
    best_node = feasible_nodes[best_idx]
    
    # Check if returning to depot is better
    # Compare the score of the best candidate with the cost of returning to depot
    # Cost of returning to depot: just dist(current, depot), but we lose the chance to visit
    # A simple heuristic: if the best candidate's score is much worse than going to depot,
    # and there are few nodes left, consider returning.
    
    # For now, always take the best feasible candidate unless no feasible nodes
    return int(best_node)

# EVOLVE-BLOCK-END
