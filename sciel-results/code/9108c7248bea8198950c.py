"""Improved CVRP heuristic: capacity-aware nearest-neighbor with return-to-depot consideration."""
import numpy as np
import time

# Module-level deadline tracking
_START_TIME = time.time()
DEADLINE_MARGIN = 2.0  # seconds of safety margin before 150s kill


def _time_exceeded():
    return (time.time() - _START_TIME) > (150.0 - DEADLINE_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next customer to visit or return to depot (0).
    
    Strategy: For each feasible unvisited customer, compute a score that balances:
    1. Distance from current node to the customer
    2. Expected cost of returning to depot after serving that customer
    3. A penalty for customers that are far from the depot (to avoid getting stranded)
    
    This is a look-ahead heuristic that considers the return trip cost.
    """
    if _time_exceeded():
        return depot  # Safety: return to depot to close route

    if len(unvisited_nodes) == 0:
        return depot

    # Filter to only feasible customers (demand fits in remaining capacity)
    feasible = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if len(feasible) == 0:
        return depot

    feasible = np.array(feasible)
    
    # Distances from current node to each feasible customer
    dist_from_current = distance_matrix[current_node, feasible]
    
    # Distances from each feasible customer back to depot
    dist_to_depot = distance_matrix[feasible, depot]
    
    # Score: distance from current + a fraction of distance back to depot
    # Using a weighted combination to balance immediate cost vs. future return cost
    # The weight on return-to-depot helps avoid picking customers that are far from depot
    # when we're running low on capacity
    weight = 0.3  # Fraction of return distance to include in score
    
    scores = dist_from_current + weight * dist_to_depot
    
    # Pick the customer with minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible[best_idx])
