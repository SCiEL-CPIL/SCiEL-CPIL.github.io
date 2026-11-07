"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _norm_cdf(x):
    """Vectorized standard normal CDF using scipy."""
    return norm.cdf(x)

def _norm_pdf(x):
    """Vectorized standard normal PDF using scipy."""
    return norm.pdf(x)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement 
    and UCB with a Gaussian cost-efficiency kernel.
    
    The acquisition score combines:
    1. Expected Improvement (EI) for exploitation
    2. A UCB-style exploration term for uncertainty
    3. A cost-efficiency weight that favors points with cost near the 
       expected minimum cost (1.5 for uniform points in [0,1]^4)
    
    This balances exploration and exploitation while being sensitive to 
    the cost structure of the problem.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive for stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning we expect improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI components
    cdf_z = _norm_cdf(z)
    pdf_z = _norm_pdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term: kappa * sigma
    # Use a moderate kappa to balance exploration
    kappa = 1.5
    ucb_term = kappa * sigma_safe
    
    # Combined potential: blend of EI and UCB
    # Weight EI more heavily as it directly measures expected gain
    blend_ratio = 0.7  # 70% EI, 30% UCB
    potential = blend_ratio * ei + (1 - blend_ratio) * ucb_term
    
    # Cost-efficiency weight using a Gaussian kernel
    # Center at expected minimum cost (1.5) with a moderate bandwidth
    # This favors points with cost near 1.5 (typical for uniform in [0,1]^4)
    cost_center = 1.5
    cost_bandwidth = 0.5
    cost_weight = np.exp(-0.5 * ((cost - cost_center) / cost_bandwidth) ** 2)
    
    # Scale the cost weight to have a reasonable range
    # At cost = 1.5, weight = 1.0
    # At cost = 1.0 or 2.0, weight = exp(-0.5 * (0.5/0.5)^2) = exp(-0.5) ≈ 0.607
    # At cost = 1.0 or 2.0 (extremes), weight ≈ 0.607
    
    # Final score: potential * cost_weight
    # This encourages points that have high potential AND low cost
    scores = potential * cost_weight
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
