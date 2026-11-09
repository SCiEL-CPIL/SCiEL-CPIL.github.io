"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviation predictions.
    f_best: float, the best observed function value so far.
    cost: (2000,) array of costs associated with evaluating each candidate.
    
    Returns:
    score: (2000,) array of acquisition scores (higher is better).
    """
    start_time = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: Clip sigma to avoid division by zero and numerical issues
    # 1e-9 is a safe floor for standard deviation in this context
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement (Phi(z)) and PDF (phi(z))
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If f_best is very large or mu is very small, EI can be large.
    # If sigma is small, EI approaches max(0, f_best - mu).
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (should be, but numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We want to maximize the expected improvement per unit cost.
    # Cost is at least 1.0, so 1/cost is a valid discount.
    # Using 1/cost directly might penalize expensive points too harshly if EI is similar.
    # However, for Rosenbrock, the minimum is at a high-cost region.
    # Let's stick to EI / cost as a robust baseline for cost-awareness.
    
    # Handle potential division by zero or negative costs (though cost >= 1)
    cost_safe = np.clip(cost, 1e-9, None)
    
    score = ei / cost_safe
    
    # Check deadline
    # The vectorized operations above are very fast (< 1ms usually).
    # If for some reason this takes too long (unlikely), we rely on the fact that it's already computed.
    # If we were doing iterative search, we'd check here.
    if time.time() - start_time > 320:
         # Fallback: return what we have. It should be valid.
         pass

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
