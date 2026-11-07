"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining expected improvement and information gain.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std dev predictions (positive)
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Normalize cost: cost is in [1, 2], map to [0, 1] for penalty
    cost_norm = (cost - 1.0) / 1.0  # [0, 1]
    
    # Adaptive parameters based on f_best
    # Early: high exploration, late: high exploitation
    f_best_safe = max(f_best, 1e-10)
    # Map f_best from [0.001, 10] to progress [0, 1]
    log_fbest = np.log(f_best_safe)
    log_min, log_max = np.log(0.001), np.log(10.0)
    progress = np.clip((log_max - log_fbest) / (log_max - log_min), 0, 1)
    # progress=0 (f_best large) -> explore, progress=1 (f_best small) -> exploit
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI is non-negative; higher means more expected improvement
    
    # Information gain proxy: sigma (uncertainty)
    # Points with high sigma provide more information about the landscape
    info_gain = sigma_safe
    
    # Combine EI and information gain with adaptive weights
    # Early: weight information gain more (explore)
    # Late: weight EI more (exploit)
    w_ei = 0.3 + 0.7 * progress  # [0.3, 1.0]
    w_info = 1.0 - w_ei          # [0.7, 0.0]
    
    # Normalize EI and info_gain to comparable scales
    # Use robust scaling to avoid issues with extreme values
    ei_scale = np.percentile(ei, 95) if np.any(ei > 0) else 1.0
    info_scale = np.percentile(info_gain, 95) if np.any(info_gain > 0) else 1.0
    ei_norm = ei / (ei_scale + 1e-10)
    info_norm = info_gain / (info_scale + 1e-10)
    
    # Combined score before cost penalty
    raw_score = w_ei * ei_norm + w_info * info_norm
    
    # Cost-aware penalty: penalize expensive evaluations
    # Use exponential penalty for stronger cost awareness
    cost_penalty = np.exp(0.5 * cost_norm)  # [1, e^0.5 ≈ 1.65]
    
    score = raw_score / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
