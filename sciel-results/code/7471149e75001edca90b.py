"""Cost-aware acquisition for Bayesian optimization (hartmann 3d).
Uses a fast, principled acquisition that combines Expected Improvement with
UCB-style exploration, cost-discounted. Parameters are set adaptively based
on the GP posterior statistics (no expensive calibration loop)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 320.0  # 320s safety margin within 330s budget


def _hybrid_acquisition(mu, sigma, f_best, cost):
    """Hybrid EI+UCB acquisition with cost discounting.
    
    score = [alpha * EI + (1-alpha) * UCB] / cost^beta
    
    where EI = (f_best - mu)*Phi(z) + sigma*phi(z), z = (f_best-mu)/sigma
          UCB = -mu + gamma*sigma
    
    alpha and gamma are chosen adaptively from the posterior statistics.
    beta is a fixed cost exponent.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component
    # Adaptive gamma: scale exploration by how uncertain we are relative to f_best
    sigma_mean = np.mean(sigma_safe)
    f_best_scale = max(abs(f_best), 1e-6)
    # gamma in [0.5, 3.0]: more exploration when sigma is large relative to f_best
    gamma = 0.5 + 2.5 * np.clip(sigma_mean / f_best_scale, 0.0, 1.0)
    ucb = -mu + gamma * sigma_safe
    
    # Blend EI and UCB
    # Use higher EI weight when we have good exploitation signal (low mu variance)
    mu_std = np.std(mu)
    mu_range = max(np.ptp(mu), 1e-6)
    # alpha in [0.3, 0.8]: favor EI more when mu is concentrated (exploit)
    alpha = 0.3 + 0.5 * np.clip(1.0 - mu_std / mu_range, 0.0, 1.0)
    
    combined = alpha * ei + (1.0 - alpha) * ucb
    
    # Cost discounting: beta=1.0 means divide by cost
    beta = 1.0
    cost_safe = np.maximum(cost, 1e-8)
    score = combined / (cost_safe ** beta)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle any non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute hybrid acquisition
    scores = _hybrid_acquisition(mu_safe, sigma_safe, f_best, cost_safe)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
