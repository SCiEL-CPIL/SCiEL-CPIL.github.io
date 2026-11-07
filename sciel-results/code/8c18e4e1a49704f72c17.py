import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Adaptive Expected Improvement (EI) where the exploration parameter xi
    scales with the current posterior uncertainty (mean sigma).
    """
    # Calculate the mean posterior uncertainty to adapt the exploration parameter
    # Using median might be more robust to outliers in the 2000 candidates
    mean_sigma = np.mean(sigma)
    
    # Adaptive xi: scales linearly with uncertainty.
    # A larger xi requires a greater potential improvement to be selected,
    # promoting exploration in high-uncertainty regions.
    # As the GP becomes more certain (sigma decreases), xi decreases,
    # shifting the focus to exploitation.
    xi = 0.1 * mean_sigma
    
    # Compute z-scores for the adjusted EI
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / (sigma + 1e-9)
    
    # Expected Improvement with xi
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if the threshold is not met; clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
