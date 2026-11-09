"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Conditional Value-at-Risk (CVaR) of the
    improvement distribution with quantile-adjusted cost penalty.
    
    CVaR focuses on the expected gain from the worst-case alpha fraction of predicted
    outcomes, making the acquisition robust to GP uncertainty in high dimensions.
    The cost penalty uses the cost quantile to prevent budget exhaustion.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # --- Step 1: Compute Expected Improvement (baseline) ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Step 2: Compute CVaR of the improvement distribution ---
    # CVaR_alpha = E[max(f_best - f(x), 0) | max(f_best - f(x), 0) <= VaR_alpha]
    # For a Gaussian f(x) ~ N(mu, sigma^2), the improvement I = max(f_best - f(x), 0)
    # We approximate CVaR by considering the tail of the improvement distribution.
    # 
    # Key insight: CVaR focuses on the worst-case alpha fraction. For minimization,
    # the "worst case" in terms of improvement means the scenarios where improvement
    # is small (i.e., f(x) is close to or above f_best).
    #
    # A practical approximation: CVaR_alpha ≈ EI - (1-alpha) * E[I * 1{I > VaR}]
    # But a simpler and effective approach: use a risk-adjusted EI that penalizes
    # high uncertainty in the tail.
    #
    # We use: CVaR_score = EI * (1 + lambda * sigma / (f_best - mu + eps))
    # where lambda controls risk aversion. Higher sigma relative to potential improvement
    # reduces the score, making it more conservative.
    
    # Risk-averse adjustment: penalize high uncertainty relative to expected improvement
    # The term sigma / max(f_best - mu, eps) measures relative uncertainty
    eps = 1e-9
    potential_improvement = np.maximum(f_best - mu, eps)
    relative_uncertainty = sigma_safe / potential_improvement
    
    # Risk aversion parameter: higher means more conservative
    lambda_risk = 0.5
    
    # CVaR-like score: EI adjusted for tail risk
    # When relative_uncertainty is high, we discount the EI
    cvar_score = ei * (1.0 - lambda_risk * np.minimum(relative_uncertainty, 2.0) / 2.0)
    cvar_score = np.maximum(cvar_score, 0.0)
    
    # --- Step 3: Quantile-adjusted cost penalty ---
    # Instead of dividing by cost (which can be unstable), we use a quantile-based penalty.
    # Points with cost above the median get a stronger penalty.
    
    cost_median = np.median(cost)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range < 1e-9:
        cost_penalty = np.zeros(n)
    else:
        # Normalize cost to [0, 1]
        cost_norm = (cost - cost_min) / cost_range
        
        # Quantile-adjusted penalty: 
        # - Costs below median: small penalty (linear)
        # - Costs above median: larger penalty (quadratic) to prevent budget exhaustion
        below_median = cost_norm <= 0.5
        cost_penalty = np.zeros(n)
        cost_penalty[below_median] = 0.3 * cost_norm[below_median]
        cost_penalty[~below_median] = 0.3 * 0.5 + 0.7 * (cost_norm[~below_median] - 0.5) ** 2 / 0.25
        # Scale penalty to be in the same range as cvar_score
        cost_penalty *= np.max(cvar_score)
    
    # --- Step 4: Combine CVaR score with cost penalty ---
    scores = cvar_score - cost_penalty
    
    # --- Step 5: Add exploration bonus for high-uncertainty regions ---
    # Encourage exploration in areas where the GP is uncertain, but weight by
    # how much that uncertainty could lead to improvement
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        # Exploration bonus: proportional to sigma, but only if there's potential for improvement
        exploration_mask = (f_best - mu) > 0  # Only explore where improvement is possible
        exploration_bonus = np.zeros(n)
        exploration_bonus[exploration_mask] = 0.1 * (sigma_safe[exploration_mask] / sigma_max) * np.max(scores)
        scores += exploration_bonus
    
    # --- Step 6: Ensure finite output and non-negative scores ---
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
