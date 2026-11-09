"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 280  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a deterministic hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration weights and soft cost penalties.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite and safe
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Time check - if close to timeout, return a fast deterministic fallback
    if not _elapsed_ok():
        improvement = f_best - mu
        z = improvement / sigma
        ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
        ei = np.maximum(ei, 0.0)
        return ei / cost

    # 1. Calculate Expected Improvement (EI)
    # EI measures the expected amount by which f_best can be improved
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Upper Confidence Bound (UCB) component
    # For minimization, we want to explore where the function *could* be low.
    # UCB = mu - kappa * sigma. We take the "optimistic" value.
    # We use a kappa that adapts to the scale of the problem.
    # If sigma is large relative to the spread of mu, we explore more.
    mu_std = np.std(mu)
    if mu_std < 1e-9:
        mu_std = 1.0
    
    # Adaptive kappa: 
    # If the uncertainty (sigma) is high relative to the mean variation (mu_std), 
    # we trust the uncertainty more and explore.
    # We normalize sigma by mu_std to get a dimensionless uncertainty measure.
    sigma_norm = sigma / mu_std
    
    # Base exploration factor
    # We want a balance. If EI is high, we exploit. If EI is low, we explore.
    # Let's use a weighted sum: score = w_ei * EI + w_ucb * (mu - kappa * sigma)
    # However, (mu - kappa * sigma) can be negative. We only care about the "potential improvement" part.
    # A better exploration term is the probability of improvement or just the uncertainty itself if normalized.
    
    # Let's stick to the successful structure of Inspiration 1 but refine the adaptive alpha.
    # Score = (EI + alpha * sigma) / cost^gamma
    
    # Determine alpha (exploration weight)
    # If the best possible EI is small compared to the average uncertainty, we are in a "flat" region or near optimum, so explore more.
    max_ei = np.max(ei)
    avg_sigma = np.mean(sigma)
    
    # If max_ei is very small, it means we are close to the optimum or the GP is confident.
    # In that case, we should rely more on sigma to find new regions.
    # If max_ei is large, we should exploit that.
    
    # Heuristic: 
    # alpha = base_alpha * (1 + uncertainty_ratio)
    # where uncertainty_ratio = avg_sigma / (max_ei + epsilon)
    
    eps = 1e-9
    uncertainty_ratio = avg_sigma / (max_ei + eps)
    
    # Clip the ratio to prevent extreme values
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.0, 10.0)
    
    # Base alpha is 0.5 (from Inspiration 1 which scored well)
    # If uncertainty is high relative to improvement, increase alpha.
    alpha = 0.5 * (1.0 + 0.5 * uncertainty_ratio)
    
    # Clip alpha to a reasonable range
    alpha = np.clip(alpha, 0.1, 3.0)
    
    # Combined raw score
    raw_score = ei + alpha * sigma
    
    # 3. Cost Penalty
    # Cost is 1 + mean(x_scaled). Range [1, 2].
    # We want to prefer cheaper points, but not exclusively.
    # Inspiration 1 used gamma=1.0. 
    # Let's try gamma=0.8 to slightly soften the cost penalty, allowing some expensive points if they have very high EI.
    # Or keep gamma=1.0 for strong cost awareness.
    # Given the budget is tight (30 units), cost awareness is crucial.
    # However, if the minimum is at a high cost location, we must be able to find it.
    # EI will be high there. If EI is high enough, it should overcome the cost penalty.
    # Let's use gamma = 1.0 as in the high-performing Inspiration 1.
    
    gamma = 1.0
    cost_factor = cost ** gamma
    
    scores = raw_score / cost_factor
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if not _elapsed_ok():
        return ei / cost
        
    return scores
# EVOLVE-BLOCK-END
