"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of GP mean predictions (shape 2000,)
    - sigma: array of GP standard deviations (shape 2000,)
    - f_best: lowest observed function value (scalar)
    - cost: array of evaluation costs (shape 2000,)
    
    Returns:
    - scores: array of acquisition values (shape 2000,), higher = evaluate next
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score: EI per unit cost, with a small bonus for low-cost points
    # This encourages finding good improvements efficiently
    base_score = ei / cost
    
    # Add a small exploration/exploration-cost tradeoff
    # Points with lower cost get a slight bonus when EI is similar
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-9
    cost_z = (cost - cost_mean) / cost_std
    
    # Mild penalty for high-cost points, reward for low-cost
    cost_adjustment = 0.05 * (-cost_z)  # negative for high cost, positive for low cost
    
    # Combine: primary signal is EI/cost, secondary is cost efficiency
    scores = base_score * (1.0 + cost_adjustment)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
