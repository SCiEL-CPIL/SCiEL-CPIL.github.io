# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) divided by cost, with a slight
    exploration boost via a small UCB term to avoid premature convergence.
    
    The score is:
        score(x) = EI(x) / cost(x) + epsilon * (f_best - mu(x)) / cost(x)
        
    EI is the standard expected improvement for minimization:
        EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    The second term provides a gentle preference for lower mean values
    (exploitation) while the EI term handles the full posterior expectation.
    """
    t_start = time.time()
    TIME_LIMIT = 320.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # --- Compute Expected Improvement (EI) for minimization ---
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu_safe) / sigma_safe
    # Clip z to avoid extreme values in normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Cost Discount ---
    # Use inverse cost as the primary penalty.
    scores = ei / cost_safe
    
    # Add a small exploitation term: prefer points with lower mu
    # This helps break ties and guides the search more decisively
    epsilon = 0.05
    exploit_term = epsilon * np.maximum(f_best - mu_safe, 0.0) / cost_safe
    scores = scores + exploit_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores

    return scores
# EVOLVE-BLOCK-END
