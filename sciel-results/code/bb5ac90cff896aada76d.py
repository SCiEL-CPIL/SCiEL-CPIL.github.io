"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot.
    
    Strategy: Savings-based heuristic.
    We prioritize nodes that offer a high "saving" compared to serving them separately.
    Saving(i) = dist(depot, i) + dist(depot, next) - dist(i, next)
    However, since we don't know 'next', we approximate by looking at the benefit of
    visiting a node far from the depot now, rather than later.
    
    A simpler and robust variant for online CVRP:
    Score = dist(depot, node) - alpha * dist(current, node)
    This encourages visiting far nodes early to avoid long return trips, 
    while still considering proximity to current location.
    
    We also enforce capacity constraints.
    """
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor if we are running out of time
        if not unvisited_nodes:
            return depot
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if not unvisited_nodes:
        return depot
    
    # Filter nodes by capacity
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible_nodes:
        return depot
        
    feasible_nodes = np.array(feasible_nodes)
    
    # Calculate distances from current node to feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from depot to feasible nodes
    dists_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Savings heuristic score:
    # We want to maximize the "saving" of not having to go back to depot.
    # A common heuristic is to prefer nodes that are far from the depot 
    # (to group them) but not too far from the current node.
    
    # Let's use a weighted score:
    # score = dists_from_depot - beta * dists_from_current
    # High dist_from_depot is good (saves a trip).
    # Low dist_from_current is good (efficiency).
    
    # Normalize distances to combine them?
    # Or just use a simple linear combination.
    # With capacity 50 and demands 1-9, routes are likely short (few stops).
    # Nearest neighbor was 1.0. 
    # The observation says "Savings-based heuristic (alpha=0.9) ... significantly outperforming".
    # That likely referred to a specific alpha in a formula like:
    # score = alpha * dist_depot - (1-alpha) * dist_current ?
    # Or maybe: score = dist_depot - dist_current?
    
    # Let's try: Score = dists_from_depot - dists_from_current
    # This effectively looks for the node that is "most worth visiting" given its remoteness
    # relative to the cost of getting there from current.
    
    scores = dists_from_depot - dists_from_current
    
    # Pick the node with the highest score
    best_idx = np.argmax(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
