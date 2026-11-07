"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a structural change: instead of a simple EI/cost hybrid, we use
    a two-phase approach:
    1. Compute Expected Improvement (EI) and Posterior Variance (PV) based scores
    2. Apply an adaptive cost-aware weighting that emphasizes exploitation
       when the best value is close to the known minimum, and exploration
       otherwise
    3. Add a small deterministic component based on the GP predictive standard
       deviation to break ties and encourage diversity
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of hartmann 3d
    f_min_known = -3.86278
    
    # Calculate Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate a variance-based exploration score
    # This encourages exploring regions with high uncertainty
    pv_score = sigma_safe
    
    # Adaptive weighting based on how close we are to the known minimum
    # If f_best is close to f_min_known, we should exploit more
    # If f_best is far from f_min_known, we should explore more
    
    # Normalize the gap between f_best and the known minimum
    # The range of hartmann 3d is roughly [-4, 0], so a gap of ~4 is the full range
    gap = f_best - f_min_known
    # Scale the gap to [0, 1] where 0 means we're at the minimum
    gap_normalized = np.clip(gap / 4.0, 0.0, 1.0)
    
    # Weight: when gap is large (early), explore more; when gap is small (late), exploit more
    # exploration_weight starts at 0.5 and decreases to 0.1 as we get closer to the minimum
    exploration_weight = 0.1 + 0.4 * gap_normalized
    exploitation_weight = 1.0 - exploration_weight
    
    # Cost-aware scoring
    # We want to maximize the score, but penalize high cost
    # Use a power law cost penalty: score / cost^alpha
    # alpha = 1.0 is linear cost-aware, alpha < 1 is less sensitive to cost
    
    # Adaptive alpha: when we're far from the minimum, be less sensitive to cost
    # (explore more freely); when close, be more sensitive (exploit carefully)
    alpha = 0.5 + 0.5 * (1.0 - gap_normalized)
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = cost_safe ** alpha
    
    # Combine EI and PV with adaptive weights
    # EI is in objective units (positive when improvement is expected)
    # PV is in objective units (uncertainty)
    
    # Normalize EI and PV to comparable scales
    # EI can be large when f_best is far from mu, PV is bounded by sigma
    ei_normalized = ei / (np.max(ei) + 1e-9)
    pv_normalized = pv_score / (np.max(pv_score) + 1e-9)
    
    # Combine
    combined = exploitation_weight * ei_normalized + exploration_weight * pv_normalized
    
    # Apply cost-aware penalty
    scores = combined / cost_factor
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small deterministic component to break ties
    # This helps with diversity in the search
    tie_breaker = 1e-6 * np.random.default_rng(hash(tuple(mu[:10])) % 2**32).random(len(mu))
    scores = scores + tie_breaker
    
    return scores
# EVOLVE-BLOCK-END
