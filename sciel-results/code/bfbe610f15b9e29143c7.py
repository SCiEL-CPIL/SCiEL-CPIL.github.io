"""Cost-aware acquisition for Ackley 2D Bayesian optimization using PI + EI blend."""
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Probability of Improvement (PI)
    with Expected Improvement (EI), using adaptive cost normalization.
    
    Strategy: PI provides smooth gradients in multimodal landscapes while EI
    drives exploitation. Cost^0.5 normalization balances exploration efficiency
    better than linear cost division.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (scalar)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        scores: array of acquisition scores, shape (2000,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Handle cases where sigma is very small to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Normalized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values that cause numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (for minimization)
    pi = norm.cdf(z_clipped)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Adaptive weighting based on f_best: as f_best approaches 0 (optimum),
    # weight PI more heavily for stability; when f_best is high, weight EI more
    # for faster exploitation
    f_best_clipped = max(f_best, 1e-6)
    w_pi = 0.5 + 0.5 * np.exp(-f_best_clipped / 0.5)
    w_ei = 1.0 - w_pi
    
    # Blend PI and EI
    blended = w_pi * pi + w_ei * ei
    
    # Use cost^0.5 normalization for better exploration efficiency
    cost_norm = np.sqrt(cost_safe)
    
    scores = blended / cost_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
