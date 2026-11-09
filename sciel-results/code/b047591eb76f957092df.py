import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines EI and UCB with adaptive weighting based on uncertainty.
    """
    t0 = time.time()
    deadline = t0 + 250.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure no division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI) for minimization
    # E[max(0, f_best - f(x))]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # Clip to non-negative

    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x). UCB is mu - beta * sigma.
    # To make it a "higher is better" score, we use -UCB = beta * sigma - mu.
    # Adaptive beta based on uncertainty level
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1.0
    
    # Higher uncertainty -> more exploration (higher beta)
    # Scale beta between 1.0 and 3.0 based on relative uncertainty
    uncertainty_ratio = sigma_safe / (sigma_mean + 1e-9)
    beta = 1.0 + 2.0 * np.clip(uncertainty_ratio, 0.0, 1.0)
    
    ucb_score = beta * sigma_safe - mu

    # Adaptive weighting: blend EI and UCB
    # When uncertainty is high, weight UCB more; when low, weight EI more
    # Use a smooth transition based on z-score magnitude
    z_abs = np.abs(z)
    # When |z| is small (high uncertainty relative to improvement), favor exploration
    # When |z| is large (low uncertainty or clear improvement), favor exploitation
    w_ucb = np.clip(1.0 / (1.0 + z_abs), 0.3, 0.7)
    w_ei = 1.0 - w_ucb

    raw_score = w_ei * ei + w_ucb * ucb_score

    # Cost-aware: divide by cost
    scores = raw_score / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if time.time() > deadline:
        return scores

    return scores
