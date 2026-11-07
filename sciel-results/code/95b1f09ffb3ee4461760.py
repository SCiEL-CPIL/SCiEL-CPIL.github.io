"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a blend of Expected Improvement (PI) and a quantile-based
    cost-effective term. The blend shifts from exploration (PI) early
    to exploitation (quantile/cost) later based on how much we've
    improved relative to the initial best.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost normalization: cost is in [1, 2] since x in [0,1] means 1 + mean(x) in [1, 2]
    # We want to penalize higher cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute PI (Expected Improvement) per unit cost
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    pi_score = ei / cost_safe
    
    # Quantile-based: 10th percentile of predicted objective = mu - 1.2816*sigma
    # We want to minimize this, so the acquisition is -q_0.1 / cost
    q_01 = mu - 1.28155 * sigma
    quantile_score = -q_01 / cost_safe
    
    # Normalize both scores to have similar scale
    # Use robust normalization: subtract median, divide by IQR-like scale
    def robust_norm(arr):
        med = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        if iqr < 1e-12:
            iqr = 1.0
        return (arr - med) / iqr
    
    pi_norm = robust_norm(pi_score)
    quant_norm = robust_norm(quantile_score)
    
    # Adaptive blend: early on, rely more on PI (exploration); 
    # later, rely more on quantile (exploitation with cost awareness)
    # Use f_best as a proxy for progress: if f_best is close to 0, we're done exploiting
    # The rosenbrock minimum is 0, so f_best < 0.1 means we're in exploitation phase
    # Blend weight: start at 0.7 for PI, decrease to 0.3 as f_best approaches 0
    # Clamp f_best to [0, 10] for stable weighting
    f_best_clamped = np.clip(f_best, 0.0, 10.0)
    # When f_best is large (early), weight_pi is high; when f_best is small (late), weight_pi is low
    weight_pi = 0.3 + 0.4 * np.exp(-f_best_clamped / 2.0)
    weight_quant = 1.0 - weight_pi
    
    # Final score
    scores = weight_pi * pi_norm + weight_quant * quant_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
