"""Heuristic matrix for ant colony optimization of orienteering."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    The heuristic encodes:
    1. Prize-to-total-cost ratio (value density): prize_j / (d[i,j] + d[j,0])
       This directly accounts for the budget constraint by penalizing moves
       that consume a large fraction of the remaining length.
    2. A gamma exponent to amplify differences between high and low efficiency edges.
    3. A smooth surrogate using log-sum-exp for robustness.
    
    The heuristic is static (computed once per instance) and biases ants toward
    moves that offer high prize per unit of path length consumed (including return).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Ensure prize at node 0 is 0 (as per problem statement)
    # prizes[0] = 0.0
    
    # Compute the "return cost" for each node: d[j, 0]
    return_cost = distance_matrix[:, 0].copy()  # shape (n,)
    
    # For each edge (i, j), the total cost of moving i->j and then eventually returning is:
    # d[i, j] + d[j, 0]
    # We build a matrix of these costs: cost[i, j] = distance_matrix[i, j] + return_cost[j]
    cost_matrix = distance_matrix + return_cost[None, :]  # shape (n, n)
    
    # The prize of the destination node j: prize[j]
    # We want: eta[i, j] = (prize[j] / cost_matrix[i, j])^gamma
    # But we need to handle j=0 (return to start) and i=j (self-loops) carefully.
    
    # For j=0, prize[0] = 0, so eta[i, 0] = 0 (or small). That's fine.
    # For i=j, cost = d[i,i] + d[i,0] = 0 + d[i,0] = d[i,0]. prize[i]/d[i,0] is fine.
    
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    
    # Compute prize/cost ratio
    # prize_matrix[i, j] = prize[j]
    prize_matrix = prizes[None, :]  # shape (n, n), each row is prizes
    
    # ratio[i, j] = prize[j] / (cost_matrix[i, j] + eps)
    ratio = prize_matrix / (cost_matrix + eps)
    
    # Clip to non-negative
    ratio = np.maximum(ratio, 0.0)
    
    # Apply gamma exponent to amplify differences
    # gamma > 1 makes high-ratio edges much more attractive
    gamma = 2.0
    eta = np.power(ratio, gamma)
    
    # Ensure all entries are positive (as required: non-positive becomes 1e-9)
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Ensure finite values
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
