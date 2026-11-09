# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Quantile-Based Lower Confidence Bound (LCB) divided by cost.
    
    The score is:
        score(x) = max(0, f_best - LCB_q(x)) / cost(x)
        
    where LCB_q(x) is the q-th quantile of the predictive distribution.
    This approach directly optimizes for expected improvement per unit cost,
    using a quantile-based uncertainty measure which is robust to outliers
    and aligns with the "Quantile-Based" directive.
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

    # --- Compute Quantile-Based LCB ---
    # Use the 10th percentile (q=0.1) as a balance between exploitation and exploration.
    # LCB_q = mu + sigma * norm.ppf(q)
    # Since we are minimizing, a lower LCB is better.
    # Improvement = f_best - LCB_q
    q = 0.1
    z_q = norm.ppf(q)  # Approx -1.28155
    
    lcb = mu_safe + sigma_safe * z_q
    improvement = f_best - lcb
    
    # Ensure improvement is non-negative for scoring (points not expected to beat f_best get 0)
    improvement = np.maximum(improvement, 0.0)
    
    # --- Cost Discount ---
    # Use inverse cost as the primary penalty, which is known to be effective.
    # To follow the directive's "sigmoidal discount" idea, we can blend or just use 1/cost.
    # Given the knowledge that EI/cost works well, and LCB is similar to EI,
    # we use improvement / cost.
    # To make it "aggressively prune" high costs, 1/cost is already convex and steep for low costs.
    # We can add a small sigmoidal component if needed, but 1/cost is simpler and robust.
    
    scores = improvement / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores

    return scores
# EVOLVE-BLOCK-END
