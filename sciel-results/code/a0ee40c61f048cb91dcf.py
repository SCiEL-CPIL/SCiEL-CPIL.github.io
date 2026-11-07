"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Uses a structural shift: cost-aware lower confidence bound (LCB) with
    adaptive exploration scaled by uncertainty, combined with a penalty
    term that discounts high-cost regions more aggressively.
    
    Parameters:
        mu: (2000,) array of GP mean predictions (standardized units)
        sigma: (2000,) array of GP std predictions (standardized units)
        f_best: scalar, best observed value (standardized units)
        cost: (2000,) array of evaluation costs (in [1, 2])
    
    Returns:
        (2000,) array of acquisition scores (higher = more desirable)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive exploration coefficient:
    # Scale kappa by the ratio of median uncertainty to mean prediction spread,
    # so exploration is stronger when the model is globally uncertain.
    med_sigma = np.median(sigma_safe)
    std_mu = np.std(mu)
    # If mu has little spread, rely more on exploration
    kappa_base = 0.3
    kappa = kappa_base * (1.0 + med_sigma / (std_mu + 1e-8))
    # Cap kappa to avoid excessive exploration
    kappa = np.clip(kappa, 0.1, 2.0)
    
    # Cost-aware LCB for minimization: lower is better
    # We want high scores for low predicted values and high uncertainty
    # Penalty: divide by cost to prefer cheaper evaluations
    lcb = mu - kappa * sigma_safe
    
    # Convert to "desirability" score: negate so higher = better
    score_base = -lcb
    
    # Cost-awareness: use a power penalty to more strongly discount expensive points
    # cost is in [1, 2]; we use cost^alpha as a divisor
    # Higher alpha means stronger cost preference
    alpha = 1.5
    cost_penalty = np.power(cost_safe, alpha)
    
    score = score_base / cost_penalty
    
    # Add a small EI component for exploitation of known good regions
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend: primarily LCB, with a small EI contribution
    # Scale EI to be comparable to score_base magnitude
    score_scale = np.std(score_base) + 1e-8
    ei_scaled = ei / (np.std(ei) + 1e-8)
    
    # Weight: 80% LCB, 20% EI
    score = 0.8 * score + 0.2 * (ei_scaled / score_scale)
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
