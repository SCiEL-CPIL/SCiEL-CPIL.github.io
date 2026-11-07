"""Cost-aware acquisition for 2D Ackley BO (pure cost-normalized EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for each candidate.
    Higher is better (means "evaluate next").

    We use expected improvement per unit cost. This is the canonical
    cost-aware acquisition function for Bayesian optimization.

    mu, sigma: arrays shape (N,)
    f_best: scalar (best observed value so far)
    cost: array shape (N,) of evaluation costs
    """
    n = mu.shape[0]
    eps = 1e-9

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=eps, posinf=1e6, neginf=eps)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    cost = np.maximum(cost, eps)
    sigma = np.maximum(sigma, eps)

    # Compute z-score for expected improvement
    improvement = f_best - mu
    z = improvement / sigma

    # Clip z for numerical stability
    z = np.clip(z, -10.0, 10.0)

    cdf = norm.cdf(z)
    pdf = norm.pdf(z)

    # Standard expected improvement
    ei = improvement * cdf + sigma * pdf

    # Cost-normalized EI
    scores = ei / cost

    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
