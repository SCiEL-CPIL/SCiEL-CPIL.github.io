# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probabilistic Expected Improvement (EI).
    
    Strategy:
    1. Compute the standard Expected Improvement (EI) for minimization.
       EI(x) = E[max(f_best - f(x), 0)]
       For a Gaussian f(x) ~ N(mu, sigma^2), this is:
       EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
       where Phi is the CDF and phi is the PDF of the standard normal distribution.
       
       Note: Since we are minimizing, 'improvement' is (f_best - f(x)).
       If f(x) < f_best, improvement is positive.
       
    2. Apply a cost-aware normalization.
       The cost of evaluation is 1 + mean(x_scaled).
       We divide the EI by the cost to maximize the expected improvement per unit cost.
       This naturally penalizes expensive evaluations (those with high x coordinates)
       unless they promise significantly higher improvement.
       
    3. Why EI over UCB?
       - UCB with a fixed kappa is sensitive to the scale of sigma.
       - EI adapts to the uncertainty: if sigma is large, EI grows roughly linearly with sigma (exploration).
       - If sigma is small, EI behaves like the deterministic improvement (exploitation).
       - EI has a built-in trade-off that is often more robust than manual kappa tuning.
       - Dividing by cost ensures we don't waste budget on high-cost points with marginal gains.
    """
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    # This represents how many standard deviations the predicted mean is below the current best.
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) = z * sigma
    # So EI = sigma * (z * Phi(z) + phi(z))
    # This form is numerically more stable and efficient.
    
    # Use scipy.stats.norm for CDF (Phi) and PDF (phi)
    # We need to handle potential numerical issues with very large/small z, 
    # but scipy's norm is generally robust.
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI calculation
    ei = sigma_safe * (z * Phi_z + phi_z)
    
    # Ensure EI is non-negative (it should be, but numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Expected Improvement per unit cost
    scores = ei / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
