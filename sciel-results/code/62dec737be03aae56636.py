import numpy as np

def acquisition(mu, sigma, f_best, cost):
    beta = 1.0
    # For minimization, we want to maximize the "optimistic" potential.
    # The potential is low mu (good) and high sigma (uncertainty).
    # Standard UCB for minimization: Score = -mu + beta * sigma
    scores = (-mu + beta * sigma) / (cost + 1e-9)
    return scores
