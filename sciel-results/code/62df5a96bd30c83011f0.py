"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a demand-aware heuristic matrix for ACO.
    
    The heuristic combines inverse squared distance with a demand penalty.
    eta[i, j] = (1 / d(i, j)^2) * (1 / (1 + demand[j]))
    
    This biases ants towards nearby nodes and away from high-demand nodes,
    which may help in packing strategies or avoiding capacity violations 
    early in the route construction, although pheromones will adjust this 
    over iterations.
    """
    n = distance_matrix.shape[0]
    
    # Ensure distances are not zero to avoid division by zero
    # Using a small epsilon
    eps = 1e-9
    safe_dist = np.maximum(distance_matrix, eps)
    
    # Inverse squared distance
    inv_dist_sq = 1.0 / (safe_dist ** 2)
    
    # Demand factor for destination nodes
    # demands is a vector of length n
    # We want to penalize high demand
    demand_factor = 1.0 / (1.0 + demands)
    
    # Broadcast demand_factor to columns
    # eta[i, j] uses demand[j]
    eta = inv_dist_sq * demand_factor[np.newaxis, :]
    
    # Ensure all entries are positive
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Ensure finite values
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return eta
# EVOLVE-BLOCK-END
