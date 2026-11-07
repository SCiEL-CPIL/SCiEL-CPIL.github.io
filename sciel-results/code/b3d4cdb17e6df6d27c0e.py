import numpy as np
from scipy.stats import norm, t as student_t
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Student's t-distribution based Expected Improvement to account for
    heavy-tailed uncertainty inherent in the limited 30-sample budget.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Use Student's t-distribution with moderate degrees of freedom
    # to account for heavy-tailed uncertainty from limited samples
    df = 5  # degrees of freedom
    
    # Calculate the t-score for the expected improvement
    # Using a small xi to balance exploitation and exploration
    xi = 0.01
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement using Student's t-distribution
    # EI_t = (f_best - mu - xi) * T_df(z) + sigma * t_pdf(z, df)
    t_cdf = student_t.cdf(z, df)
    t_pdf = student_t.pdf(z, df)
    
    ei_t = (f_best - mu - xi) * t_cdf + sigma_safe * t_pdf
    
    # Add a UCB-like exploration term to encourage exploring high-uncertainty regions
    # This helps escape local minima in the 6D space
    # The coefficient is chosen to be small so that EI dominates in low-uncertainty regions
    ucb_boost = 0.05 * sigma_safe
    
    # Combine EI and UCB boost
    scores = ei_t + ucb_boost
    
    # Ensure all returned scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
