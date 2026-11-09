"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    The heuristic should guide ants towards nodes with high prize and low distance.
    We want to maximize prize collection within the max_len constraint.
    
    Key considerations:
    1. Prize at node 0 is 0, so we shouldn't be attracted to node 0 (except as return)
    2. Distance matters - closer nodes are easier to visit and return from
    3. We need to be able to return to node 0, so d[i,j] + d[j,0] matters
    4. Higher heuristic = more attractive
    
    Strategy: Use prize / (distance from current + distance to return)
    This naturally favors high-prize nodes that are close and easy to return from.
    """
    # Make sure we work with numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Create a matrix where entry [i, j] represents attractiveness of moving from i to j
    # We want to favor:
    # - High prize at j
    # - Low distance from i to j
    # - Low distance from j back to 0 (so we can still return)
    
    # Prize component: prize[j] for each target j
    # Distance component: distance_matrix[i, j] + distance_matrix[j, 0] (cost to visit j and return)
    
    # Create the heuristic matrix
    # For each pair (i, j), heuristic[i, j] = prize[j] / (dist[i,j] + dist[j,0] + epsilon)
    
    epsilon = 1e-9
    
    # dist_to_return[j] = distance_matrix[j, 0] - cost to return from j to start
    dist_to_return = distance_matrix[:, 0]  # shape (n,)
    
    # total_cost[i, j] = distance_matrix[i, j] + dist_to_return[j]
    total_cost = distance_matrix + dist_to_return[None, :]  # shape (n, n)
    
    # heuristic[i, j] = prize[j] / (total_cost[i, j] + epsilon)
    # This gives higher values for high-prize, low-cost moves
    
    heuristic_matrix = prizes[None, :] / (total_cost + epsilon)
    
    # Ensure all values are positive (task says non-positive become 1e-9)
    heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
    
    # Ensure finite values
    heuristic_matrix = np.where(np.isfinite(heuristic_matrix), heuristic_matrix, 1e-9)
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
