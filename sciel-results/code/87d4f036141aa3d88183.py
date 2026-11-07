"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using improvement density per unit cost.
    Models the search as maximizing the density of the improvement distribution
    at the threshold (phi(z)) normalized by evaluation cost.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    f_best_val = float(f_best)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    # z = (f_best - mu) / sigma
    z = (f_best_val - mu) / sigma_safe
    
    # The improvement density at the threshold is phi(z), the standard normal PDF
    # This represents the density of the improvement distribution at f_best
    phi_z = norm.pdf(z)
    
    # Primary score: improvement density per unit cost
    # This prioritizes points where the improvement distribution has high density
    # at the threshold (high-confidence improvements) with low cost
    density_score = phi_z / cost_safe
    
    # Secondary component: expected improvement per unit cost for exploitation
    Phi_z = norm.cdf(z)
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    ei_score = ei / cost_safe
    
    # Tertiary component: exploration via uncertainty per unit cost
    explore_score = sigma_safe / cost_safe
    
    # Adaptive weighting based on how close f_best is to the minimum (0 for Griewank)
    # When f_best is small (near optimum), focus on exploitation (EI and density)
    # When f_best is large (far from optimum), focus on exploration (sigma)
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    
    # Weights: density is primary, EI secondary, exploration tertiary
    # As we get closer to the optimum, increase weight on density and EI
    w_density = 0.5 + 0.3 * (1.0 - f_best_clipped)
    w_ei = 0.3 * (1.0 - f_best_clipped) + 0.1
    w_explore = 0.2 * f_best_clipped
    
    # Normalize weights
    w_sum = w_density + w_ei + w_explore
    w_density /= w_sum
    w_ei /= w_sum
    w_explore /= w_sum
    
    scores = w_density * density_score + w_ei * ei_score + w_explore * explore_score
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple improvement density per cost
        scores = phi_z / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
