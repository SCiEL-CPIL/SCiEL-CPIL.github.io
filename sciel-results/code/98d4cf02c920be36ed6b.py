"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
DEADLINE = START_TIME + 110.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next customer to visit using a refined heuristic.
    
    The heuristic minimizes the estimated marginal increase in route length:
    d(current, candidate) + d(candidate, depot)
    
    This accounts for both the travel to the candidate and the potential cost
    of returning to the depot, encouraging compact routes and avoiding
    unnecessary extensions away from the depot.
    """
    # Check deadline
    if time.time() > DEADLINE:
        return 0

    # Convert unvisited_nodes to a numpy array
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return 0

    # Filter by capacity
    # demands is indexed by node id
    # We need demands[node] <= rest_capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited[feasible_mask]
    
    # If only one feasible node, return it (or depot if we want to close early? No, must visit if possible)
    # Actually, if we have capacity and unvisited nodes, we should visit one.
    
    # Calculate distances from current_node to all feasible nodes
    d_current = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from depot to all feasible nodes
    d_depot = distance_matrix[depot, feasible_nodes]

    # Score: Minimize d(current, node) + d(node, depot)
    # This is the exact marginal cost if the route were to end at 'node'.
    # It serves as a strong proxy for minimizing total route length.
    scores = d_current + d_depot

    # Find the index of the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
