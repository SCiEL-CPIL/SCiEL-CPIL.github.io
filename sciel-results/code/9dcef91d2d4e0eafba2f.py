"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Log-UCB for exploration in uncertain regions
    - Normalized by cost for cost-awareness
    - Adaptively blends between EI and UCB based on uncertainty
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Primary: Probability of significant improvement (lower-bound surrogate) ---
    # Maximize P(f(x) < f_best - eps) = Phi((f_best - eps - mu) / sigma)
    # This focuses on high-confidence improvements over a fixed threshold
    typical_scale = max(abs(f_best), 0.1)
    eps = 0.05 * typical_scale
    target = f_best - eps
    
    z_prob = (target - mu) / sigma_safe
    z_prob_clipped = np.clip(z_prob, -50, 50)
    prob_improve = norm.cdf(z_prob_clipped)
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # --- Secondary: Expected Improvement for exploitation ---
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Normalize components to [0, 1]
    prob_max = np.max(prob_improve) if np.max(prob_improve) > 0 else 1.0
    prob_norm = prob_improve / prob_max
    
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Blend: primarily use probability of improvement, with EI as secondary signal
    # Weight probability more heavily since it's a more direct lower-bound surrogate
    w_prob = 0.7
    w_ei = 0.3
    
    combined = w_prob * prob_norm + w_ei * ei_norm
    
    # Cost-awareness: divide by cost with power < 1 to be less aggressive
    scores = combined / np.power(cost_safe, 0.7)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
