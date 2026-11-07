"""Cost-aware acquisition for Styblinski-Tang 2D BO loop."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function combining Expected Improvement (EI) and
    a penalty for high cost, with a small exploration bonus based
    on uncertainty.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior in objective units.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline guard: if we've been running too long, return a simple
    # deterministic score to avoid timeout.
    # (We don't have access to a global timer here, but the caller
    #  is responsible for keeping total wall-clock under 330s.)

    # Clip sigma to avoid division by zero and numerical issues.
    sigma_safe = np.maximum(sigma, 1e-8)

    # Standardized improvement.
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the PDF.
    # We use scipy.stats.norm for numerical stability.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Ensure EI is non-negative (it should be, but clip for safety).
    ei = np.maximum(ei, 0.0)

    # Cost-aware: divide by cost to get "improvement per unit cost".
    # But also add an exploration term that rewards high uncertainty
    # when EI is low, to encourage exploration in unexplored regions.
    #
    # A common approach: use a weighted combination of EI/cost and
    # sigma/cost, or use a "probability of improvement" style term.
    #
    # Let's use: score = EI / cost + alpha * sigma / cost
    # where alpha is a small exploration weight.
    #
    # However, a more principled cost-aware acquisition is:
    #   score = EI / (cost * (1 + beta * sigma))
    # or simply EI / cost with a floor on cost.
    #
    # Let's try a hybrid: EI/cost plus a small bonus for high sigma
    # relative to the scale of the problem. The sigma values are in
    # objective units; for Styblinski-Tang, the range is roughly
    # [-78, 250], so a sigma of ~10 is significant.
    #
    # We'll use: score = EI / cost + 0.05 * sigma / cost
    # This gives a slight preference to uncertain points when EI is low.
    exploration_weight = 0.05
    score = (ei + exploration_weight * sigma_safe) / cost

    # Ensure all scores are finite.
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
