"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 330.0
_MARGIN = 15.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Probability of Improvement (PI) with adaptive exploration.
    
    The core idea:
    - Use EI as the base improvement metric.
    - Add a PI term to encourage exploration in areas with high uncertainty 
      but also decent potential.
    - Scale the entire score by a factor that balances exploitation vs exploration
      based on the current state (though we lack global state, we can use local features).
    - Divide by cost to penalize expensive points.
    
    Structural change from previous versions:
    Instead of a fixed multiplicative boost on EI, we combine EI and PI additively 
    with a dynamic weight that depends on the normalized uncertainty (sigma / (sigma + epsilon)).
    This creates a smoother transition between exploitation and exploration.
    """
    
    # Deadline guard
    if _remaining_time() <= 0:
        return np.zeros_like(mu)
    
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    cost_safe = np.maximum(cost, epsilon)
    
    # Calculate z-score for EI and PI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Probability of Improvement
    pi = cdf_z
    
    # Adaptive weighting between EI and PI
    # When sigma is large relative to the improvement potential, we want more exploration (PI)
    # When sigma is small, we want more exploitation (EI)
    # Normalize sigma to [0, 1] using a soft saturation
    sigma_norm = sigma_safe / (sigma_safe + 1.0)
    
    # Weight for EI: decreases with uncertainty
    # Weight for PI: increases with uncertainty
    # Use a sigmoid-like transition
    weight_ei = 1.0 - 0.5 * sigma_norm
    weight_pi = 0.5 * sigma_norm
    
    # Combined acquisition score
    base_score = weight_ei * ei + weight_pi * pi * f_best
    
    # Additional exploration boost for points with very high uncertainty
    # This helps escape local optima
    high_uncertainty_mask = sigma_safe > np.percentile(sigma_safe, 95)
    boost_factor = np.ones_like(base_score)
    boost_factor[high_uncertainty_mask] = 1.5
    
    score = (base_score * boost_factor) / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
