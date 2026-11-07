"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - beta * sigma(x).
    # We want to select x that minimizes L(x).
    # The evaluator selects the candidate with the HIGHEST score.
    # So we return S(x) = -L(x) = beta * sigma(x) - mu(x).
    
    beta = 2.0
    scores = beta * sigma - mu
    
    # Ensure finite scores
    # mu and sigma from GP should be finite, but just in case
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
