"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Lagrangian relaxation of cost constraint:
    # score = (f_best - mu) / cost - lambda * cost
    # The first term is expected improvement per unit cost (exploitation)
    # The second term penalizes expensive evaluations (exploration via cost avoidance)
    # lambda is dynamically tuned to balance the budget
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI per unit cost minus a penalty for high cost
    # This decouples the function landscape exploration from explicit division
    # The lambda term encourages finding good solutions at low cost
    # Use a moderate lambda that balances exploitation and cost efficiency
    lambda_cost = 0.1
    
    score = ei / cost - lambda_cost * cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
