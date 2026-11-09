"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Knowledge Gradient (KG) inspired approach:
    - Estimate the probability that each candidate is a new global minimum
    - Estimate the expected improvement in f_best if it is
    - Combine with cost awareness
    
    Parameters:
    -----------
    mu : array of shape (n,) - GP mean predictions
    sigma : array of shape (n,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (n,) - evaluation costs
    
    Returns:
    --------
    score : array of shape (n,) - acquisition values (higher = better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute the probability that each candidate has a value below f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    prob_new_min = norm.cdf(z)
    
    # Expected improvement if this point is the new minimum:
    # E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    # For a truncated normal, E[f(x) | f(x) < f_best] = mu - sigma * phi(z)/Phi(z)
    # So E[f_best - f(x) | f(x) < f_best] = (f_best - mu) + sigma * phi(z)/Phi(z)
    # And the unconditional expected improvement is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    phi_z = norm.pdf(z)
    # Avoid division by zero
    phi_over_phi_z = np.where(prob_new_min > 1e-10, phi_z / np.maximum(prob_new_min, 1e-10), 0.0)
    
    # Expected improvement (standard EI for minimization)
    ei = (f_best - mu) * prob_new_min + sigma * phi_z
    
    # KG-inspired: weight by probability of being new minimum
    # This emphasizes exploration in uncertain regions where new minima may be hidden
    kg_weight = prob_new_min * (1.0 + 0.5 * phi_over_phi_z)
    
    # Combine EI with KG weight
    # The KG term rewards points with high probability of being new minima
    # and high expected reduction in f_best
    kg_score = ei * kg_weight
    
    # Add a small exploitation bias to help navigate flat regions
    # Points with lower predicted means are slightly preferred
    exploitation_bias = -0.05 * mu
    
    # Cost-aware scoring: divide by cost to prefer cheaper evaluations
    # Use a soft cost penalty to avoid extreme penalties
    cost_penalty = cost ** 0.5  # Square root to soften cost penalty
    
    score = (kg_score + exploitation_bias) / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Normalize to prevent numerical issues
    max_score = np.max(score)
    if max_score > 0:
        score = score / max_score
    
    return score
# EVOLVE-BLOCK-END
