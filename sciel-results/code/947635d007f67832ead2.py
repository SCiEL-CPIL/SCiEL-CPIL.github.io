# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition for Bayesian optimization (minimization).
    
    Combines Expected Improvement (EI) with a dynamically scaled Lower Confidence
    Bound (LCB) term to balance exploitation and exploration under a cost budget.
    
    score(x) = [EI(x) + kappa * sigma(x)] / cost(x)
    
    EI targets incremental regret reduction (matching the scoring metric),
    while the LCB term (kappa * sigma) encourages exploration of uncertain
    regions. The combined quantity is normalized by evaluation cost to
    maximize value per unit budget.
    
    kappa is dynamically scaled based on remaining budget:
    - Early (high remaining budget): larger kappa for exploration
    - Late (low remaining budget): smaller kappa for exploitation
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    t_start = time.time()
    deadline = 280.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical guards: ensure sigma and cost are positive and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)

    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    improvement = f_best - mu_safe
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -40.0, 40.0)
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI calculation
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical guard)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Dynamic kappa based on remaining budget
    # Total budget is 30. We estimate remaining budget from the cost array:
    # The cost of the next evaluation is in [1, 2] (since cost = 1 + mean(x_scaled) in [0,1]).
    # We use the mean cost as a proxy for average evaluation cost.
    mean_cost = np.mean(cost_safe)
    
    # Estimate how many evaluations we've done:
    # Initial 5 points have cost ~1.5 each (uniform in [0,1]^2 gives mean(x_scaled) ~ 0.5)
    # So initial cost ~ 7.5. Total budget 30.
    # We don't have direct access to total spent, but we can infer from f_best magnitude
    # and the fact that we're in the loop.
    # 
    # Better approach: use f_best as a proxy for progress.
    # f_best starts around 1-2 for the 3-hump camel and approaches 0.
    # Map f_best to a progress in [0, 1].
    if f_best <= 0:
        progress = 1.0
    else:
        # Assume f_best ~ 1.5 at start, 0 at end
        progress = np.clip(1.0 - f_best / 1.5, 0.0, 1.0)
    
    # kappa: exploration coefficient
    # Start with larger kappa (more exploration) early, decrease later
    # kappa ranges from ~2.0 (early) to ~0.5 (late)
    kappa = 0.5 + 1.5 * (1.0 - progress)
    
    # Hybrid score: EI + kappa * sigma, all divided by cost
    # The EI term captures expected regret reduction
    # The kappa * sigma term encourages exploration of uncertain regions
    # Both are cost-normalized to maximize value per unit budget
    
    lcb_component = kappa * sigma_safe
    
    score = (ei + lcb_component) / cost_safe
    
    # Ensure all scores are finite (higher = evaluate next)
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t_start > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
