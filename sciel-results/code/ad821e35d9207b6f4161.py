"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization.
    # We aim to minimize the objective function.
    # The UCB for the lower bound of the function value is mu - beta * sigma.
    # We maximize the score: -(mu - beta * sigma) / cost.
    # Beta is chosen to balance exploration and exploitation.
    # A value around 2.0-2.5 is typical for early-to-mid stages of BO.
    beta = 2.5
    scores = -(mu - beta * sigma) / cost
    return scores
# EVOLVE-BLOCK-END
