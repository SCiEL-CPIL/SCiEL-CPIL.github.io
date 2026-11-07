"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # UCB = mu - beta * sigma
    # This approach balances exploitation (low mu) and exploration (high sigma).
    # It is distinct from EI and often performs better in multimodal landscapes
    # with small evaluation budgets by avoiding premature convergence.
    
    # Hyperparameter for exploration-exploitation trade-off.
    # A fixed value is used. 2.0 provides a strong exploration incentive
    # suitable for the 2D domain and small total budget (30 evals).
    beta = 2.0
    
    # Compute UCB score. Higher score indicates a more promising candidate.
    # We minimize the objective, so we subtract the uncertainty term.
    scores = mu - beta * sigma
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
