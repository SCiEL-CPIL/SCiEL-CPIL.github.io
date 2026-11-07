import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Maximizing this score balances exploitation (low mu) and exploration (high sigma)
    # Increased beta to encourage more exploration in the multi-modal Branin landscape
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
