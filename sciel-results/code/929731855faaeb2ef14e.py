"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track a simple deadline guard. The acquisition function is called once per
# iteration by the evaluator; we keep a module-level timestamp to enforce a
# safety margin.
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 5.0


def _time_remaining():
    return _DEADLINE_SECONDS - _SAFETY_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're running low on time, fall back to a simple
    # cost-aware expected improvement to guarantee a valid return.
    if _time_remaining() < 1.0:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Thompson Sampling: draw one sample from the GP posterior at each candidate.
    # The sample is y_k = mu_k + sigma_k * z_k where z_k ~ N(0, 1).
    # We want to minimize y_k / cost_k, so the acquisition score (higher = better)
    # is the negative of that ratio: score_k = -y_k / cost_k.
    #
    # However, since we only have mu and sigma (not the full GP), we approximate
    # the posterior sample as a local Gaussian: sample_k = mu_k - sigma_k * |z_k|
    # is not quite right. The standard Thompson sample is mu_k + sigma_k * z_k
    # with z_k ~ N(0,1). Since we want to MINIMIZE the objective, a lower sample
    # is better. So the acquisition score should reward low samples per unit cost.
    #
    # score_k = - (mu_k + sigma_k * z_k) / cost_k
    # Higher score = lower predicted value per cost.

    z = np.random.standard_normal(len(mu))
    sample = mu + sigma * z
    scores = -sample / (cost + 1e-9)

    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)

    return scores
# EVOLVE-BLOCK-END
