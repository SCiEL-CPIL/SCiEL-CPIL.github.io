"""Rewritten program: cost-aware acquisition with Student-t robustness."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as t_dist


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using expected improvement with
    Student-t distribution to account for model uncertainty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Student-t adjustment for robustness
    # Use degrees of freedom that reflect our uncertainty about the model
    # Lower df means heavier tails, more robust to outliers
    # We'll use a moderate df (e.g., 5-10) to balance robustness and efficiency
    df = 7.0  # Degrees of freedom for Student-t
    
    # Compute Student-t based improvement
    # The Student-t CDF and PDF scale differently than normal
    # We want to capture the idea that with lower confidence (higher sigma),
    # we should be more conservative in our improvement estimate
    
    # t-distribution CDF and PDF
    t_cdf = t_dist.cdf(z, df)
    t_pdf = t_dist.pdf(z, df)
    
    # Scale factor: t-distribution has variance df/(df-2) for df > 2
    # This means the t-distribution is "wider" than the normal
    # We use this to adjust the EI estimate
    scale_factor = np.sqrt(df / (df - 2.0)) if df > 2 else 1.0
    
    # Robust EI using Student-t
    # The idea: when sigma is large (uncertain), the t-distribution's heavier
    # tails mean we should discount the expected improvement
    robust_ei = (f_best - mu) * t_cdf + sigma_safe * t_pdf * scale_factor
    
    # Combine standard EI and robust EI
    # Weight based on uncertainty: more weight on robust version when sigma is large
    # Use a smooth blend
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    uncertainty_weight = sigma_safe / (sigma_max + 1e-9)
    
    # Blend: 50% standard EI, 50% robust EI, weighted by uncertainty
    # When uncertainty is high, favor the robust version
    blend_weight = 0.5 + 0.5 * uncertainty_weight  # ranges from 0.5 to 1.0
    blended_ei = (1 - 0.5 * uncertainty_weight) * ei + 0.5 * uncertainty_weight * robust_ei
    
    # Alternative: use a simpler robust score
    # Score = EI / (cost * (1 + sigma_penalty))
    # where sigma_penalty accounts for model uncertainty
    
    # Cost-aware scoring: divide by cost, but also penalize high uncertainty
    # The penalty should be moderate to avoid over-discounting
    uncertainty_penalty = 1.0 + 0.5 * (sigma_safe / (np.mean(sigma_safe) + 1e-9))
    
    # Final score: robust EI divided by cost and uncertainty penalty
    scores = blended_ei / (cost * uncertainty_penalty)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (EI should be non-negative when mu < f_best)
    # But due to numerical issues, clip to a small negative value
    scores = np.maximum(scores, -1e-6)
    
    return scores
# EVOLVE-BLOCK-END
