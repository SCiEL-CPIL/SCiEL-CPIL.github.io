"""Rewritten cost-aware acquisition for Ackley 2D Bayesian optimization."""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of expected improvement
    and a UCB-style exploration term, both normalized by cost.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (scalar)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        scores: array of acquisition scores, shape (2000,)
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the CDF and phi is the PDF of the standard normal
    
    # Handle cases where sigma is very small to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Normalized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # Clip z to avoid extreme values that cause numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB term for exploration: -mu + beta * sigma
    # Adapt beta based on the overall uncertainty level
    mean_sigma = np.mean(sigma)
    beta = 0.5 + 1.5 * np.exp(-mean_sigma / 0.3)
    
    ucb = -mu + beta * sigma
    
    # Blend EI and UCB, both normalized by cost
    # Weight EI more heavily as we get closer to the optimum
    # Use a fixed weight of 0.7 for EI and 0.3 for UCB
    ei_score = ei / cost_safe
    ucb_score = ucb / cost_safe
    
    scores = 0.7 * ei_score + 0.3 * ucb_score
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
