# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Quantile-Based Lower Confidence Bound (LCB) with sigmoidal cost discount.
    
    The score is:
        score(x) = max(0, f_best - LCB_q(x)) * sigmoid_discount(cost(x))
        
    where LCB_q is the q-th quantile of the predictive distribution,
    and sigmoid_discount provides a smooth, aggressive penalty for high-cost candidates.
    """
    t_start = time.time()
    TIME_LIMIT = 320.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # --- Dynamic Quantile Selection ---
    # Use a quantile that adapts to the cost distribution.
    # Lower-cost candidates should get more exploration (lower quantile).
    # Compute the median cost as a reference.
    cost_median = np.median(cost_safe)
    
    # Map cost to a dynamic quantile: lower cost -> lower quantile (more exploration)
    # cost in [1, 2], map to q in [0.05, 0.25]
    cost_normalized = (cost_safe - 1.0) / max(1e-6, 2.0 - 1.0)  # Normalize to [0, 1]
    cost_normalized = np.clip(cost_normalized, 0.0, 1.0)
    
    # Dynamic quantile: cheaper points get lower quantile (more aggressive exploration)
    q = 0.05 + 0.20 * cost_normalized
    z_q = norm.ppf(q)  # Vector of z-values
    
    # --- Compute Quantile-Based LCB ---
    lcb = mu_safe + sigma_safe * z_q
    improvement = f_best - lcb
    
    # Ensure improvement is non-negative for scoring
    improvement = np.maximum(improvement, 0.0)
    
    # --- Sigmoidal Cost Discount ---
    # Use a sigmoid to create a smooth, aggressive penalty for high-cost candidates.
    # discount = 1 / (1 + exp(k * (cost - cost_ref)))
    # where cost_ref is around the median, and k controls steepness.
    
    cost_ref = cost_median
    k = 8.0  # Steepness parameter - higher means more aggressive pruning
    
    # Sigmoid discount: close to 1 for low cost, close to 0 for high cost
    sigmoid_arg = k * (cost_safe - cost_ref)
    # Clip to avoid overflow in exp
    sigmoid_arg = np.clip(sigmoid_arg, -50, 50)
    discount = 1.0 / (1.0 + np.exp(sigmoid_arg))
    
    # Combine improvement with cost discount
    scores = improvement * discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores

    return scores
# EVOLVE-BLOCK-END
