"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Structural Change: Potential-based Heuristic
    # Instead of combining distances in the denominator with powers,
    # we define a "net value" for each node j that accounts for the 
    # cost of returning to the start (d[j, 0]).
    # This decouples the step cost d[i, j] from the global return cost.
    
    # 1. Calculate Node Potential
    # prize[j] is the gain. d[j, 0] is the "debt" of being far from start.
    # We subtract a fraction of the return distance from the prize.
    # lambda controls how heavily we penalize return distance.
    # max_len is 3. Distances are in unit square (max ~1.41).
    # Prizes are [0, 1].
    # If d[j,0] is 1.41, and lambda=0.5, penalty is 0.7.
    # A prize of 0.8 becomes 0.1. A prize of 1.0 becomes 0.3.
    # This encourages high prizes but discourages nodes that force a long return.
    
    lambda_return = 0.5
    d_j0 = distance_matrix[:, 0] # Distance from each node to start
    node_potential = prizes - lambda_return * d_j0
    
    # Ensure potentials are not negative (floor to a small positive number)
    # This ensures that even "bad" nodes have some minimal attractiveness
    # so the ant doesn't get stuck if all others are visited/invalid.
    node_potential = np.maximum(node_potential, 1e-6)
    
    # 2. Calculate Step Attractiveness
    # The cost to move from i to j is d[i, j].
    # We use inverse distance as the heuristic for the step.
    # d[i, j] is 0 if i == j. We handle this by adding epsilon or masking.
    # The ACO logic prevents moving to visited nodes, but the heuristic matrix
    # is computed for all pairs. We should make diagonal 0 or very small.
    
    d_ij = distance_matrix
    
    # Avoid division by zero on diagonal
    # We use a small epsilon
    eps = 1e-6
    step_cost = d_ij + eps
    
    # Heuristic: Potential of destination / Step Cost
    # Shape: node_potential is (n,), step_cost is (n, n)
    # We broadcast node_potential across rows.
    heuristic = node_potential[None, :] / step_cost
    
    # Ensure finite and positive
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Zero out the diagonal explicitly to prevent self-loops if pheromone is high
    # (Though ACO logic should handle visited nodes, it's safer for the heuristic)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
