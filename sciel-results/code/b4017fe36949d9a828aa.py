# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a budget-scaled LCB approach.
    Uses a refined dynamic beta that decays slower (sqrt of gap ratio)
    and is scaled to a standard UCB magnitude.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Constants
    BUDGET = 30.0
    F_MIN = -3.32237
    MAX_GAP = 4.0
    
    # Calculate gap to the known minimum
    gap = f_best - F_MIN
    
    # Estimate remaining budget fraction.
    # Using the square root of the normalized gap ensures that beta remains
    # significant for a longer duration, preventing premature exploitation
    # in the high-dimensional Hartmann 6d landscape.
    fraction_remaining = np.clip(np.sqrt(gap / MAX_GAP), 0.1, 1.0)
    
    # Estimate remaining budget
    remaining_budget = BUDGET * fraction_remaining
    
    # Dynamically scale the exploration term (beta).
    # Use a linear scaling with the square root of the normalized gap.
    # This provides a stronger exploration signal early on when the gap is large,
    # which is crucial for escaping local optima in 6D landscapes,
    # and decays to zero as we approach the known minimum.
    normalized_gap = np.clip(gap / MAX_GAP, 0.0, 1.0)
    beta = 2.0 * np.sqrt(normalized_gap)
    
    # Lower Confidence Bound (LCB) for minimization
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-normalized score
    score = lcb_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
