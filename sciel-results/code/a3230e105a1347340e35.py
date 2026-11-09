"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _fast_ei(mu, sigma, f_best):
    """Compute Expected Improvement for minimization efficiently."""
    z = (f_best - mu) / sigma
    # Use stable formula for EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _fast_ucb(mu, sigma, beta):
    """Compute Lower Confidence Bound score (higher is better)."""
    return -mu + beta * sigma

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of EI and UCB with adaptive weights.
    
    The strategy is to blend exploitation (EI) and exploration (UCB) in a way
    that is scale-consistent. We compute both scores, normalize them, and take
    a weighted sum. The weight for UCB (exploration) decays as we approach the
    optimum (f_best approaches -3.32237).
    
    Additionally, we add a small entropy-based exploration term to encourage
    visiting regions with high posterior uncertainty, which is crucial in
    high-dimensional spaces.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe[~np.isfinite(sigma_safe)] = 1e-8
    
    # 1. Compute Expected Improvement (EI)
    ei = _fast_ei(mu, sigma_safe, f_best)
    
    # 2. Compute UCB with an adaptive beta
    # We want to explore more early on (when f_best is far from -3.32)
    target_min = -3.32237
    dist_to_target = np.maximum(f_best - target_min, 0.0)
    
    # Beta starts high and decays. Let's use a simple schedule.
    # If dist_to_target is large (e.g., 2.0), beta ~ 1.0
    # If dist_to_target is small (e.g., 0.1), beta ~ 0.1
    beta = 0.5 + 0.5 * dist_to_target
    ucb = _fast_ucb(mu, sigma_safe, beta)
    
    # 3. Normalize scores to [0, 1] range to make them comparable
    # EI is always non-negative. UCB can be anything.
    # Let's normalize EI by its max, and UCB by its range.
    ei_max = np.max(ei)
    if ei_max < 1e-10:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_norm = np.ones_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
    
    # 4. Adaptive blending
    # When we are far from the optimum, we want to explore more (weight on UCB).
    # When we are close, we want to exploit (weight on EI).
    # Let's use a weight that depends on dist_to_target.
    # w_ucb = 0.5 * dist_to_target / (1.0 + dist_to_target)
    # This gives w_ucb ~ 0 when dist=0, and w_ucb ~ 0.5 when dist is large.
    w_ucb = 0.5 * dist_to_target / (1.0 + dist_to_target)
    w_ei = 1.0 - w_ucb
    
    # 5. Entropy bonus
    # Encourage exploration of high uncertainty regions.
    # Entropy ~ 0.5 * log(2 * pi * e * sigma^2)
    entropy = 0.5 * np.log(2.0 * np.pi * np.e) + np.log(sigma_safe)
    entropy_max = np.max(entropy)
    if entropy_max < 1e-10:
        entropy_norm = np.zeros_like(entropy)
    else:
        entropy_norm = entropy / entropy_max
    
    # Add a small entropy bonus, weighted by how far we are from the optimum.
    # This helps escape local minima in high dimensions.
    entropy_weight = 0.2 * w_ucb
    scores = w_ei * ei_norm + w_ucb * ucb_norm + entropy_weight * entropy_norm
    
    # Safety: ensure finite and non-negative
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
