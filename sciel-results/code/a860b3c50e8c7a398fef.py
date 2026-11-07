"""Rewritten acquisition function using cost-aware LCB with dynamic exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Lower Confidence Bound (LCB) normalized by cost, with a dynamic
    exploration coefficient that adapts based on the current state of the
    optimization.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard: if we're running out of time, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: just use -mu/cost
        scores = -mu / cost
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Dynamic exploration coefficient
    # Early on: explore more (higher beta), later: exploit more (lower beta)
    # Use the ratio of f_best to the scale of mu to determine progress
    mu_scale = np.max(np.abs(mu)) if np.any(np.abs(mu) > 1e-10) else 1.0
    progress = min(1.0, max(0.0, f_best / (mu_scale + 1e-10)))
    
    # Beta starts high (exploration) and decreases (exploitation)
    # Using a smooth schedule
    beta = 2.0 * (1.0 - progress) + 0.5 * progress
    
    # Lower Confidence Bound for minimization:
    # LCB = mu - beta * sigma
    # We want to minimize this, so higher LCB scores are better
    # But we need to convert to an acquisition score where higher is better
    # Since we want low LCB values (good), we negate: score = -LCB
    # But we also want to be cost-aware
    
    # LCB score (lower is better for the objective)
    lcb = mu - beta * sigma
    
    # For acquisition, we want high scores. Since lower LCB is better,
    # we can use -lcb as a base score, but we need to normalize by cost.
    # Actually, let's think differently:
    # The expected improvement in the objective from evaluating x is roughly
    # max(0, f_best - mu) but LCB gives a more optimistic view.
    # A common approach: score = (f_best - lcb) / cost
    # This rewards points where the LCB is much lower than f_best (potential improvement)
    # and penalizes high cost.
    
    # Alternative: use the LCB directly, normalized
    # score = -lcb / cost
    # But this can be negative and may not properly rank candidates.
    
    # Better: use expected improvement with a confidence-bound flavor
    # EI_LCB = max(0, f_best - (mu - beta * sigma))
    # But this can be overly optimistic. Let's use a smoothed version.
    
    # Actually, let's use a hybrid: combine EI with LCB
    # EI for minimization:
    safe_sigma = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / safe_sigma
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma * phi
    ei = np.maximum(ei, 0.0)
    
    # LCB-based optimistic improvement:
    # The LCB gives a lower bound on the true function value
    # So the optimistic improvement is max(0, f_best - lcb)
    lcb_improvement = np.maximum(0.0, f_best - lcb)
    
    # Blend EI and LCB improvement based on uncertainty
    # When uncertainty is high, trust LCB more; when low, trust EI more
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        uncertainty_ratio = sigma / sigma_max
    else:
        uncertainty_ratio = np.zeros_like(sigma)
    
    # Weight: higher uncertainty -> more weight on LCB
    w_lcb = 0.3 + 0.4 * uncertainty_ratio  # range [0.3, 0.7]
    w_ei = 1.0 - w_lcb
    
    combined_improvement = w_ei * ei + w_lcb * lcb_improvement
    
    # Normalize by cost
    scores = combined_improvement / cost
    
    # Add a small exploration bonus for high-uncertainty, low-cost points
    explore_bonus = 0.02 * sigma / cost
    
    scores = scores + explore_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
