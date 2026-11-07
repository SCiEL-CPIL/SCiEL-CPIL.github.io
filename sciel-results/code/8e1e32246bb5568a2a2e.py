"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Strategy:
    1. Compute the standard Expected Improvement (EI) for minimization.
       EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma, Phi is the CDF of the standard normal,
       and phi is the PDF of the standard normal.
       
    2. Normalize the EI by the cost of evaluation to get the expected improvement
       per unit cost. This encourages finding improvements in cheap regions 
       (low cost) while still accounting for the magnitude of potential improvement.
       
    3. This deterministic approach avoids the high variance of Thompson Sampling,
       which can lead to erratic exploration. EI provides a smoother landscape 
       that balances exploitation (high EI near known good points) and exploration 
       (high EI in high uncertainty regions) more consistently.
       
    4. We handle edge cases where sigma is very small to avoid numerical issues.
    """
    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for the improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, we look for points where mu < f_best.
    # The formula above is standard for maximization of improvement (f_best - f).
    # If mu > f_best, z is negative, Phi(z) is small, phi(z) is small.
    # If mu < f_best, z is positive, Phi(z) is large, phi(z) is large.
    # This correctly weights the potential gain.
    
    # Use scipy.stats.norm for stable CDF and PDF calculations
    # norm.cdf(z) gives Phi(z)
    # norm.pdf(z) gives phi(z)
    
    # Vectorized calculation
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    # Cost-aware score: EI divided by cost
    # We add a small epsilon to cost to prevent division by zero, though cost >= 1
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
