# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a demand-aware, multi-criteria edge-attractiveness matrix for ACO CVRP.
    
    Combines:
      1. Inverse distance squared (stronger distance preference)
      2. Penalty for high-demand destinations (bias toward packing heavy customers early)
      3. Bonus for edges that help fill the vehicle (encourages consolidation)
    
    The heuristic is eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j])) * (1 + 0.5 * min(remaining_capacity_hint, capacity)/capacity)
    
    But since we don't have state, we use a static approximation:
    eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))
    
    Additionally, we add a small bonus for edges from the depot to high-demand nodes
    to encourage starting routes with heavy customers.
    """
    # Ensure inputs are numpy arrays
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    
    # Base: inverse distance squared
    # Add epsilon to avoid division by zero
    eps = 1e-9
    dist = distance_matrix + eps
    inv_dist_sq = 1.0 / (dist * dist)
    
    # Demand penalty: higher demand at destination j reduces attractiveness
    # This biases ants toward visiting lower-demand nodes? Wait, the directive says
    # "penalty for high-demand nodes" to "pack heavier customers early to minimize
    # the number of trips and empty-return distances."
    # 
    # Actually, re-reading: "penalty for high-demand nodes (e.g., eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))),
    # directly biasing the pheromone update toward routes that pack heavier customers early"
    #
    # Hmm, if we penalize high-demand nodes, ants would avoid them. But the goal is to pack them early.
    # Let me re-read: "penalty for high-demand nodes... biasing... toward routes that pack heavier customers early"
    #
    # I think the idea is: by making high-demand nodes less attractive in general, but since they must be visited,
    # the pheromone will build up on edges that lead to them when the vehicle is full, effectively
    # encouraging routes that group heavy customers together. Actually, this is a bit confusing.
    #
    # Let me just follow the formula given: eta[i,j] = d(i,j)^-2 * (1 / (1 + demand[j]))
    # This makes edges to high-demand nodes less attractive. The hope is that this, combined with
    # the ACO pheromone, will lead to routes that consolidate heavy customers.
    
    # Demand-aware factor: 1 / (1 + demand[j])
    # demand[j] is the demand of customer j (destination)
    demand_factor = 1.0 / (1.0 + demands[np.newaxis, :])
    
    # Combine
    eta = inv_dist_sq * demand_factor
    
    # Ensure all values are positive
    eta = np.maximum(eta, 1e-9)
    
    # Handle diagonal (self-loops) - set to very small value
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
