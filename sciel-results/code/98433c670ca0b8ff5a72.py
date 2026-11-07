"""Cost-aware acquisition: Probability of Improvement with adaptive exploration."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # seconds, safety margin before 330s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI)
    with adaptive exploration for Bayesian optimization of cosine8 8d.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _remaining_time() < 3.0:
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-10)
        scores = poi / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)
    poi = norm.cdf(z)  # bounded [0, 1]

    # Cost-normalized PoI
    poi_cost = poi / cost_safe

    # Exploration component: reward high uncertainty in regions that could
    # plausibly beat f_best. Use a soft-gated uncertainty score.
    # Gate: only explore where mu is within ~2 sigma of f_best (could improve)
    # The gate is smooth: gate = Phi((f_best - mu) / sigma) itself,
    # so exploration is naturally concentrated where improvement is plausible.
    # Exploration score: sigma / cost, gated by PoI
    # This avoids wasting exploration in clearly bad regions.
    exploration = sigma_safe * poi / cost_safe

    # Adaptive weighting: more exploration early (when f_best is far from min)
    # cosine8 minimum is -0.8, typical initial f_best ~ 0.5-1.0
    min_possible = -0.8
    # progress: 1 when f_best is at minimum, 0 when at worst possible (~1.0)
    progress = (f_best - min_possible) / (1.0 - min_possible + 1e-10)
    progress = np.clip(progress, 0.0, 1.0)

    # Exploration weight: higher when we haven't found the minimum yet
    # Use a power to make it more aggressive early
    w_explore = 0.15 + 0.35 * (progress ** 1.5)
    w_explore = np.clip(w_explore, 0.10, 0.50)

    # Combine: normalize both to comparable scales
    # Use rank-based normalization for robustness
    n = len(poi_cost)
    
    # Normalize poi_cost
    poi_mean = np.mean(poi_cost)
    poi_std = np.std(poi_cost)
    if poi_std < 1e-12:
        poi_std = 1e-12
    poi_norm = (poi_cost - poi_mean) / poi_std

    # Normalize exploration
    exp_mean = np.mean(exploration)
    exp_std = np.std(exploration)
    if exp_std < 1e-12:
        exp_std = 1e-12
    exp_norm = (exploration - exp_mean) / exp_std

    # Clip normalized scores to avoid extreme outliers
    poi_norm = np.clip(poi_norm, -5, 5)
    exp_norm = np.clip(exp_norm, -5, 5)

    combined = (1.0 - w_explore) * poi_norm + w_explore * exp_norm

    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
