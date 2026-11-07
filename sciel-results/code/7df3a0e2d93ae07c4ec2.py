# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 3D.

    Uses a hybrid of Expected Improvement (EI) and a cost-aware UCB-like term.
    The EI component drives exploitation towards points with high predicted improvement,
    while a small UCB component encourages exploration in high-uncertainty regions.
    A cost penalty is applied to prefer cheaper evaluations when other factors are equal.

    Parameters
    ----------
    mu : array-like, shape (n,)
        GP mean predictions.
    sigma : array-like, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Standard normal z-score
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB component: encourages exploration in high-uncertainty regions
    # beta controls the trade-off between exploitation and exploration
    beta = 0.2
    ucb = mu + beta * sigma_safe

    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units
    # We combine them with a weighted average
    w_ei = 0.8
    w_ucb = 0.2
    base_score = w_ei * ei + w_ucb * (-ucb)  # Negate UCB since lower is better for minimization

    # Cost penalty: subtract lambda * cost
    # The cost is in [1, 2] since it's 1 + mean(x) where x in [0,1]^3
    # A small lambda helps guide the search toward cheaper regions
    lambda_cost = 0.05
    scores = base_score - lambda_cost * cost

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
