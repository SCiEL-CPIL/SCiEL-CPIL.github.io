import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Maximizing this score balances exploitation (low mu) and exploration (high sigma)
    beta = 1.0
    scores = -mu + beta * sigma
    return scores
