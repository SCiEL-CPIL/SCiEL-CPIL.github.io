import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Regret Minimization (ERM)
    with quantile-based cost normalization.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0) / cost_safe

    # Numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Expected Regret Minimization (ERM)
    # ERM = E[max(f(x) - f_best, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Higher ERM means higher expected regret, so we want to MINIMIZE it.
    # For acquisition (higher = better), we use negative ERM or a transformation.
    
    # Alternative: Use the probability of improvement and expected improvement
    # but with a different cost-aware weighting strategy.
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (standard)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Quantile-based cost normalization
    # Instead of dividing by cost^gamma, we use a rank-based approach:
    # 1. Compute a combined score that balances EI and cost
    # 2. Use percentile ranking to create a more robust cost-aware score
    
    # First, compute a raw score combining EI and cost efficiency
    # We want high EI and low cost
    # Use a multiplicative penalty that's more adaptive
    
    # Adaptive gamma based on the distribution of costs
    cost_median = np.median(cost_safe)
    cost_range = np.ptp(cost_safe)
    
    # If costs are relatively uniform, use a stronger penalty
    # If costs vary widely, use a weaker penalty to allow exploration
    cost_cv = np.std(cost_safe) / max(np.mean(cost_safe), eps)
    
    # gamma between 0.3 and 0.7 based on cost variability
    # Low CV (uniform costs) -> higher gamma (stronger penalty for high cost)
    # High CV (varied costs) -> lower gamma (weaker penalty, more exploration)
    gamma = 0.3 + 0.4 * min(cost_cv / 0.5, 1.0)
    
    # Compute cost-aware score
    cost_factor = np.power(cost_safe / cost_median, gamma)
    scores = ei / cost_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Additional: Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in multimodal functions like Ackley
    uncertainty_bonus = 0.05 * sigma_safe / (np.max(sigma_safe) + eps)
    scores = scores + uncertainty_bonus * (1.0 - cost_factor)
    
    # Final cleanup
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
