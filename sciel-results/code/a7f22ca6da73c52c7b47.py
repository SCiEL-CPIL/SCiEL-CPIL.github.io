"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a budget-aware expected utility heuristic matrix for ant colony optimization.
    
    The heuristic is:
        eta[i, j] = (pi_j + eps) / (d_ij + d_j0 + eps)^1.5 * exp(-lambda * (d_ij + d_j0) / L_max)
    
    where:
        pi_j is the prize at node j
        d_ij is the distance from i to j
        d_j0 is the distance from j back to depot (node 0)
        L_max is max_len
        lambda is a penalty parameter (set to 1.0)
        eps is a small epsilon to avoid division by zero
    
    This penalizes moves that consume a disproportionate fraction of the remaining length budget.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    
    # Ensure prizes[0] is 0 (depot has no prize)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # Compute the "cost" of visiting node j from any node i and returning to depot:
    # cost[i, j] = d[i, j] + d[j, 0]
    # d[j, 0] is the same for all i, so we can compute it as a vector
    d_to_depot = distance_matrix[:, 0]  # shape (n,)
    d_from_any = distance_matrix  # shape (n, n)
    
    # cost[i, j] = d[i, j] + d[j, 0]
    cost = d_from_any + d_to_depot[None, :]  # shape (n, n)
    
    # Budget-aware expected utility:
    # eta[i, j] = (pi_j + eps) / (cost[i, j] + eps)^1.5 * exp(-lambda * cost[i, j] / L_max)
    eps = 1e-9
    lam = 1.0
    
    # Avoid negative or zero costs
    cost_safe = np.maximum(cost, eps)
    
    # Prize vector (broadcast over rows)
    prize_vec = prizes[None, :] + eps  # shape (1, n)
    
    # Compute the heuristic
    # The exponential decay penalizes moves that use a large fraction of the budget
    decay = np.exp(-lam * cost_safe / max_len)
    base = prize_vec / (cost_safe ** 1.5)
    
    eta = base * decay
    
    # For the diagonal (i == j), the ant can't move to itself, but we set it to a small value
    # Actually, the ant only moves to unvisited nodes, and node 0 is the start.
    # We should set eta[i, i] to a very small value to discourage self-loops
    np.fill_diagonal(eta, eps)
    
    # Also, the ant starts at node 0. From node 0, it can't go back to node 0.
    # But the heuristic is used for all pairs. Let's make sure eta[0, 0] is small.
    # Already handled by fill_diagonal.
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    # Clamp to avoid extreme values
    eta = np.clip(eta, eps, 1e6)
    
    return eta
# EVOLVE-BLOCK-END
