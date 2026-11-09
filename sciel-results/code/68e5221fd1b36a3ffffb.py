"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is a decision-theoretic criterion that maximizes the expected value
    of the improvement over the current best observation. It is equivalent
    to a 1-step Knowledge Gradient.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - f(x).
    # Z represents how many standard deviations f_best is above the mean.
    Z = (f_best - mu) / sigma_safe
    
    # Calculate the standard normal CDF and PDF
    # Phi(Z) = CDF(Z)
    # phi(Z) = PDF(Z)
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If (f_best - mu) is negative (mu < f_best), the first term is negative.
    # However, the second term is always positive.
    # EI is always non-negative.
    
    EI_scores = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Handle numerical issues where EI might be slightly negative due to floating point errors
    # or if sigma is effectively zero and mu > f_best (which shouldn't happen if f_best is min)
    # But strictly, EI >= 0.
    EI_scores = np.maximum(EI_scores, 0.0)
    
    # Ensure finite scores
    EI_scores = np.where(np.isfinite(EI_scores), EI_scores, 0.0)
    
    return EI_scores
# EVOLVE-BLOCK-END
