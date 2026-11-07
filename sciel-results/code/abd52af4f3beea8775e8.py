"""Rewritten program using Upper Confidence Bound (UCB) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to maximize the score, where higher score means "more promising".
    # Promising points have low mu (expected value) and high sigma (uncertainty).
    # Score = -mu + beta * sigma
    
    # Tuning parameter beta controls the exploration/exploitation trade-off.
    # A value around 2.0 to 3.0 is common for small budgets.
    beta = 2.0
    
    # Calculate UCB score
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
