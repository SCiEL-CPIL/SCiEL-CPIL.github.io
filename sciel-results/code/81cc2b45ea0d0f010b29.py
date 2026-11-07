"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) acquisition scores for minimization.
    
    Strategy: 
    1. Calculate raw EI assuming zero exploration offset (xi=0).
    2. Estimate an exploration offset (xi) from the distribution of raw EI values.
       We use a quantile of the positive EI values to encourage exploration 
       beyond the immediate best prediction, helping escape local optima 
       in the multimodal Hartmann 3D landscape.
    3. Recalculate EI with the chosen xi.
    4. Ensure outputs are finite.
    """
    # Input validation and edge cases
    if mu.size == 0:
        return np.array([])
    
    # Handle case where sigma is effectively zero or negative (numerical issues)
    # sigma should be standard deviation, so non-negative.
    sigma = np.maximum(sigma, 1e-12)
    
    # Standard Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # Where Phi is CDF and phi is PDF of standard normal.
    
    # To avoid importing scipy.stats for speed and simplicity, we can use 
    # numpy's erf for Phi and exp for phi, or just use scipy if available.
    # The prompt says SciPy is available. Using scipy.stats.norm is robust.
    
    from scipy.stats import norm
    
    # Initial pass with xi = 0 to estimate scale of improvement
    z0 = (f_best - mu) / sigma
    ei0 = (f_best - mu) * norm.cdf(z0) + sigma * norm.pdf(z0)
    
    # Filter for positive improvements to estimate a reasonable exploration offset.
    # If all EI are <= 0, the model thinks current best is likely the global min, 
    # or we are in a weird state. In that case, fall back to UCB-like behavior 
    # or small xi.
    positive_ei = ei0[ei0 > 1e-8]
    
    if len(positive_ei) > 0:
        # Use the 90th percentile of positive EI as a target for exploration.
        # This encourages points that are "significantly" better than the best 
        # observed, or in regions where the model is uncertain enough to 
        # predict such an improvement.
        # Actually, a common heuristic is to use a quantile of the EI distribution.
        # Let's try using the 75th percentile of the positive EI values as xi.
        # This makes the acquisition function favor points with EI > xi.
        xi = np.percentile(positive_ei, 75)
    else:
        # If no positive EI, use a small default based on sigma scale
        xi = 0.01 * np.mean(sigma)
        
    # Ensure xi is not too large
    xi = max(xi, 1e-6)
    
    # Calculate final EI with the chosen xi
    # The term becomes (f_best - xi - mu)
    z = (f_best - xi - mu) / sigma
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is typically non-negative. If it's negative, it means the expected 
    # improvement is less than xi. We can clip to 0 or keep as is.
    # Higher is better. Clipping to 0 is standard.
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
