"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def _norm_cdf(x):
    """Fast normal CDF using scipy."""
    return norm.cdf(x)


def _norm_pdf(x):
    """Fast normal PDF using scipy."""
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a structural change: instead of a fixed blend of EI/UCB/PI,
    we use a two-stage approach:
    1. Compute a "promise" score based on expected improvement with
       adaptive epsilon that scales with the current uncertainty landscape.
    2. Apply a cost-aware penalty that is more aggressive in regions where
       the GP is already confident (low sigma) but has not found improvement,
       and more lenient in high-uncertainty regions.
    
    Key structural differences from prior versions:
    - Adaptive epsilon based on the distribution of sigma values
    - Cost penalty that depends on local uncertainty (sigma) not just cost
    - Uses a "regret-weighted" EI that downweights points where the
      probability of any improvement is very low
    
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
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Step 1: Compute adaptive epsilon ---
    # Scale epsilon based on the typical uncertainty in the landscape
    # Use the median sigma as a measure of typical uncertainty
    sigma_median = np.median(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Epsilon should be small enough to capture meaningful improvements
    # but large enough to filter out noise-level improvements
    # Scale with the range of mu to make it adaptive
    mu_range = np.max(mu) - np.min(mu) if np.max(mu) != np.min(mu) else 1.0
    
    # Adaptive epsilon: proportional to the smaller of sigma_median and mu_range
    eps = 0.1 * min(sigma_median, abs(mu_range) * 0.1 + 1e-8)
    eps = max(eps, 1e-6)  # Floor to avoid zero epsilon
    
    target = f_best - eps
    
    # --- Step 2: Compute probability of significant improvement ---
    z_prob = (target - mu) / sigma_safe
    z_prob_clipped = np.clip(z_prob, -50, 50)
    prob_improve = _norm_cdf(z_prob_clipped)
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # --- Step 3: Compute expected improvement with adaptive epsilon ---
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Standard EI
    ei = delta * _norm_cdf(z_clipped) + sigma_safe * _norm_pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Regret-weighted EI: multiply by probability of meaningful improvement
    # This downweights points where even if EI is high, the probability
    # of actually beating f_best - eps is low
    regret_weighted_ei = ei * prob_improve
    
    # --- Step 4: Compute exploration bonus ---
    # In high-uncertainty regions, we should be more willing to explore
    # even if the expected improvement is low
    # Use a normalized uncertainty measure
    sigma_norm = sigma_safe / (sigma_max + 1e-8)
    
    # Exploration bonus: higher when sigma is high and mu is not too bad
    # We want to explore where there's uncertainty AND the mean is promising
    mu_norm = (mu - np.min(mu)) / (np.max(mu) - np.min(mu) + 1e-8)
    mu_promising = 1.0 - mu_norm  # Higher when mu is lower (better)
    
    exploration_bonus = sigma_norm * mu_promising
    
    # --- Step 5: Combine scores ---
    # Normalize each component
    rwe_max = np.max(regret_weighted_ei) if np.max(regret_weighted_ei) > 0 else 1.0
    rwe_norm = regret_weighted_ei / rwe_max
    
    eb_max = np.max(exploration_bonus) if np.max(exploration_bonus) > 0 else 1.0
    eb_norm = exploration_bonus / eb_max
    
    # Blend: weight exploitation more when we're confident, exploration more when uncertain
    # Use the average sigma as a global confidence measure
    avg_sigma_ratio = np.mean(sigma_safe) / (sigma_max + 1e-8)
    
    # When average uncertainty is high, weight exploration more
    w_explore = 0.3 + 0.4 * avg_sigma_ratio
    w_exploit = 1.0 - w_explore
    
    combined = w_exploit * rwe_norm + w_explore * eb_norm
    
    # --- Step 6: Cost-aware penalty ---
    # The cost penalty should be more aggressive in regions where the GP
    # is already confident (low sigma) but has not found improvement
    # In high-uncertainty regions, we can afford to pay more cost
    
    # Local confidence: 1 - sigma_norm (high when sigma is low)
    local_confidence = 1.0 - sigma_norm
    
    # Cost penalty: higher cost penalty when confidence is high
    # Use a power law with adaptive exponent
    cost_exponent = 0.5 + 0.5 * local_confidence  # Range [0.5, 1.0]
    
    cost_penalty = np.power(cost_safe, cost_exponent)
    scores = combined / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
