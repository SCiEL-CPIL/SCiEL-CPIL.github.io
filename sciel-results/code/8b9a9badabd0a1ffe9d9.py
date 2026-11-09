"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) strategy.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean predictions from GP.
    sigma : array_like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x_scaled_to_01)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher is better.
    """
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values in inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate an adaptive exploration parameter beta.
    # We use the mean of the top 10% of sigma values to scale the exploration term,
    # ensuring that beta adapts to the current uncertainty landscape.
    # This is more robust than a fixed constant.
    if len(sigma) > 0:
        # Use the 90th percentile of sigma as a representative uncertainty scale
        sigma_scale = np.percentile(sigma, 90)
        if not np.isfinite(sigma_scale) or sigma_scale < 1e-9:
            sigma_scale = 1.0
        # Base beta is 2.0, scaled slightly by the magnitude of uncertainty
        # This helps maintain a balance: higher uncertainty -> higher absolute exploration step
        beta = 2.0 * (1.0 + 0.1 * np.log1p(sigma_scale))
    else:
        beta = 2.0
        
    # LCB for minimization is usually mu - beta * sigma.
    # Since we need to MAXIMIZE the acquisition score (higher is better),
    # we negate the LCB: score_unscaled = -(mu - beta * sigma) = -mu + beta * sigma.
    # This means points with low mu (good candidates) and high sigma (uncertainty) get high scores.
    scores_unscaled = -mu + beta * sigma
    
    # Cost-aware acquisition: score per unit cost
    scores = scores_unscaled / cost
    
    # Ensure all scores are finite. 
    # If a score is negative, it's still valid as long as it's finite, but 
    # typically we expect positive scores for "good" candidates. 
    # However, the relative ranking is what matters.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace any -inf with a very small number to ensure "finite" requirement if strict,
    # though the prompt says "finite scores". -inf is not finite.
    # Let's map -inf to a very low value, e.g., -1e10
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
