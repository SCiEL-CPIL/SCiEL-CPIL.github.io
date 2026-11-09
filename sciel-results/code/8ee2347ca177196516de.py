"""Cost-aware Bayesian Regret Minimization acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Bayesian Regret Minimization.

    Directly targets the scoring metric (simple regret) by approximating
    the expected reduction in the minimum observed value.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 1e10)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        # Fallback: simple cost-aware LCB
        score = (f_best - mu + 0.5 * sigma) / cost
        return np.where(np.isfinite(score), score, -1e10)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Bayesian Regret Minimization:
    # E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    # The regret reduction is the term subtracted, so we maximize it.
    
    # Compute the improvement potential with a conservative uncertainty penalty
    improvement_potential = f_best - mu - 0.5 * sigma_safe
    
    # The expected regret reduction is proportional to max(0, improvement_potential)
    # But we also want to account for the uncertainty: high sigma means high potential
    # for a large improvement, but also high risk.
    
    # A better approximation: E[max(0, f_best - f(x))] where f(x) ~ N(mu, sigma^2)
    # This is the Expected Improvement, but we want to target the minimum directly.
    
    # Let's use a hybrid: the core idea is that we want to minimize the expected
    # minimum after evaluation. E[min(f_best, f(x))] = f_best - E[max(0, f_best - f(x))]
    # So maximizing the regret reduction = maximizing E[max(0, f_best - f(x))]
    
    # This is exactly EI, but the directive suggests a different form. Let's use:
    # regret_reduction = 0.5 * max(0, f_best - mu - 0.5*sigma)
    # This is a conservative estimate that accounts for uncertainty.
    
    regret_reduction = 0.5 * np.maximum(0.0, f_best - mu - 0.5 * sigma_safe)
    
    # Additionally, add a small exploration bonus for high uncertainty regions
    # where the GP is uncertain, to encourage exploration in early stages
    # Scale the exploration bonus by how far f_best is from 0 (early = more exploration)
    exploration_weight = np.clip(f_best / 5.0, 0.0, 1.0)
    exploration_bonus = exploration_weight * 0.3 * sigma_safe
    
    # Combine: primary signal is regret reduction, secondary is exploration
    score = (regret_reduction + exploration_bonus) / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
