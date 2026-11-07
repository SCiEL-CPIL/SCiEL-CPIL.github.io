"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    D = np.asarray(distance_matrix, dtype=float)
    P = np.asarray(prizes, dtype=float)
    
    # d_0i: distance from start (node 0) to each node i. Shape (n, 1)
    d_0i = D[0, :].reshape(-1, 1)
    
    # d_j0: distance from each node j to start (node 0). Shape (1, n)
    d_j0 = D[:, 0].reshape(1, -1)
    
    # d_ij: distance matrix. Shape (n, n)
    d_ij = D
    
    # Structural change: Use marginal gain model
    # Instead of a fixed geometric ratio, estimate the "value density" of adding node j
    # to a path that has just visited i, considering the remaining budget.
    
    # The remaining budget after moving i -> j and returning j -> 0 is:
    # remaining = max_len - d_ij - d_j0
    # This is the budget available for future nodes.
    # However, the ant is at i, so it has already spent some budget to get to i.
    # A better proxy for "opportunity cost" is how much of the *total* budget is consumed
    # by the transition i->j->0 relative to the potential value.
    
    # Let's try a "Prize Density" approach:
    # Attractiveness of j from i is proportional to Prize[j] / (distance cost of j).
    # But we also want to penalize nodes that are too far from the start or too far from i
    # such that they leave little room for other nodes.
    
    # New Model:
    # H[i, j] = P[j] * (1 + slack_factor) / (d_ij + eps)
    # Where slack_factor encourages moves that leave more budget.
    
    # Let's look at the directive: "weighting prize[j] by this slack"
    # slack = max_len - d[i,j] - d[j,0]
    # If slack is high, it means we can still visit many other nodes.
    # If slack is low, we are close to the limit.
    
    # Let's try: H[i, j] = P[j] * max(0, slack) / (d_ij + eps)
    # This directly implements the directive.
    
    eps = 1e-9
    slack = max_len - d_ij - d_j0
    
    # Ensure slack is non-negative for the heuristic calculation (infeasible moves get 0 weight here)
    positive_slack = np.maximum(slack, 0.0)
    
    # Base term: Prize / Distance
    base = P[None, :] / (d_ij + eps)
    
    # Weight by slack to prioritize moves that conserve budget
    H = base * positive_slack
    
    # Ensure all values are positive and finite.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
