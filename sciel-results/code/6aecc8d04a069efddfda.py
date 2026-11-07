import numpy as np
from scipy.stats import norm
import time

# Cost-aware acquisition function for Bayesian optimization (levy 2d)
# Strategy: Diversity-aware cost-normalized Expected Improvement.
# We penalize candidates that are in "crowded" regions (low sigma) to encourage
# exploration of unexplored areas, while still rewarding high expected improvement
# relative to evaluation cost.

# Track call count to adapt exploration
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware cost-normalized acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    global _call_count
    _call_count += 1
    
    # Compute z-score for Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Diversity penalty: penalize candidates in low-uncertainty (crowded) regions.
    # We use a hyperbolic penalty that is large when sigma is small (crowded)
    # and small when sigma is large (unexplored).
    # The penalty scales with the inverse of sigma, but is capped to avoid numerical issues.
    # We also scale the penalty by a factor that decays over time to allow more exploitation later.
    # Early on, we want strong diversity; later, we can afford more exploitation.
    diversity_weight = 1.0 / (1.0 + 0.02 * _call_count)
    
    # Crowding penalty: higher when sigma is lower.
    # Use 1/sigma_safe as a proxy for "density" of the GP's confidence.
    # But to avoid extreme values, we can use a softer penalty like 1/(sigma_safe + epsilon).
    # Let's use a penalty proportional to 1/sigma_safe, but normalized to the scale of EI.
    # A simpler approach: subtract a term that is large when sigma is small.
    # Let's define penalty = diversity_weight * (1.0 / (sigma_safe + 1e-3))
    # But this might be too strong. Let's try a log-based penalty or a scaled inverse.
    # Alternatively, we can use the fact that low sigma implies the region is well-explored.
    # So we subtract a term that is proportional to the "confidence" (1/sigma).
    # Let's use: penalty = diversity_weight * (1.0 / (sigma_safe + 1e-2))
    # This encourages moving to regions with higher sigma (uncertainty).
    
    # To balance the scales, we can normalize the penalty by the mean EI or something similar.
    # But for simplicity, let's use a fixed scale.
    # Let's try: penalty = diversity_weight * (1.0 / (sigma_safe + 1e-2))
    # And then score = ei / cost - penalty / cost
    
    # Actually, let's use a more robust penalty:
    # penalty = diversity_weight * np.exp(-sigma_safe)  # This is small when sigma is large, large when sigma is small? No, exp(-small) is ~1, exp(-large) is ~0.
    # So exp(-sigma) is high when sigma is small (crowded). That's what we want.
    # Let's use: penalty = diversity_weight * np.exp(-sigma_safe)
    # This is bounded between 0 and diversity_weight, which is good for numerical stability.
    
    penalty = diversity_weight * np.exp(-sigma_safe)
    
    # Cost-aware: divide both EI and penalty by cost
    cost_safe = np.maximum(cost, 1e-9)
    score = (ei - penalty) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
