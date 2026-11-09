"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and PI for minimization.
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    # Expected Improvement (EI) for minimization
    # Improvement = f_best - mu (positive when mu < f_best)
    improvement = f_best - mu
    
    # Standardized improvement
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    z = improvement / sigma_safe
    
    # EI for minimization: E[max(0, f_best - f)]
    # = sigma * (z * phi(z) + Phi(z)) where phi is standard normal PDF, Phi is CDF
    from scipy.stats import norm
    ei = sigma * (z * norm.pdf(z) + norm.cdf(z))
    
    # Probability of Improvement (PI) for minimization
    # P(f < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Blend EI and PI
    # Weight by (improvement + sigma) to balance exploitation and exploration
    weight = np.maximum(improvement + sigma, 1e-10)
    
    # Normalize EI and PI to have similar scales
    # EI is in objective units, PI is in [0, 1]
    # Scale PI by typical sigma to match EI scale
    typical_sigma = np.median(sigma)
    pi_scaled = pi * typical_sigma
    
    # Blend: 0.7 * EI + 0.3 * PI_scaled
    blended = 0.7 * ei + 0.3 * pi_scaled
    
    # Make cost-aware by dividing by cost
    epsilon = 1e-8
    cost_safe = cost + epsilon
    
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
