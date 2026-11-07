import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling with exploitation bias.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The best (lowest) objective value observed so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Thompson Sampling: sample from posterior predictive distribution
    # For minimization, we want low values, so score = -(mu - sigma * z) = -mu + sigma * z
    # where z ~ N(0,1). Higher score means more likely to be a good (low) value.
    
    # Generate standard normal samples
    z = np.random.standard_normal(mu.shape)
    
    # Thompson sampling score: -mu + sigma * z
    # This represents the negative of a sample from the posterior (since we minimize)
    ts_score = -mu + sigma * z
    
    # Add exploitation bias toward the known best value
    # The bias encourages exploration of regions near f_best when uncertainty is low
    # and allows more exploration when uncertainty is high
    # Use a scaled version of the expected improvement as a bias term
    
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    Z = (f_best - mu) / sigma_safe
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Blend Thompson sampling with a deterministic exploitation component
    # The blend weight can be adjusted; here we use a fixed blend
    # Higher weight on TS for exploration, lower for exploitation
    blend_weight = 0.7
    score = blend_weight * ts_score + (1 - blend_weight) * ei
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
