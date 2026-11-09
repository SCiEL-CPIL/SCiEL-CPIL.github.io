"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_start_time = None


def acquisition(mu, sigma, f_best, cost):
    """
    Dynamic Expected Improvement with cost-aware exploration budget.
    
    Uses EI as the core acquisition, with a dynamic exploration weight that
    scales by the ratio of estimated remaining budget to current candidate cost.
    Early iterations (high uncertainty, more budget) explore more; late iterations
    (low uncertainty, less budget) exploit more.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _acq_start_time
    
    # Deadline guard: if we've been running too long, return safe fallback
    if _acq_start_time is not None:
        elapsed = time.time() - _acq_start_time
        if elapsed > 280:
            return -mu.copy()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Guard against zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Dynamic exploration budget estimation ---
    # Total budget is 30. We estimate remaining budget by looking at the
    # distribution of costs and the current state of the GP.
    # 
    # Key insight: the ratio of sigma to the scale of f_best tells us how
    # much we've learned. Early on, sigma is large relative to f_best, so
    # we should explore. Late on, sigma is small, so we should exploit.
    
    # Estimate the "learning progress" from the GP predictions
    # When mu is tightly clustered around f_best, we're close to convergence
    mu_centered = mu - f_best
    mu_std = np.std(mu_centered) if np.std(mu_centered) > 1e-9 else 1.0
    sigma_mean = np.mean(sigma) if np.mean(sigma) > 1e-9 else 1.0
    
    # Learning ratio: high when we're still uncertain (early), low when certain (late)
    # Normalize by the scale of the problem
    problem_scale = max(abs(f_best), 1.0)
    learning_ratio = sigma_mean / (mu_std + 1e-9)
    
    # Clamp learning ratio to a reasonable range
    # Early: learning_ratio can be large (sigma >> mu spread)
    # Late: learning_ratio approaches 0 or 1
    learning_ratio = np.clip(learning_ratio, 0.1, 5.0)
    
    # --- Cost-aware exploration weight ---
    # The exploration weight should be:
    # 1. Higher for cheaper candidates (lower cost)
    # 2. Higher when we have more relative budget (early iterations)
    # 3. Lower when we're close to convergence (late iterations)
    
    mean_cost = np.mean(cost_safe)
    # Cost efficiency: 1.0 for average cost, >1 for cheap, <1 for expensive
    cost_efficiency = mean_cost / cost_safe
    
    # Dynamic beta: combines learning progress and cost efficiency
    # Base exploration weight scales with learning ratio
    # Modulated by cost efficiency so cheap points get more exploration
    base_beta = 1.0 + 0.5 * learning_ratio
    beta = base_beta * cost_efficiency
    
    # --- Expected Improvement for minimization ---
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are the standard normal CDF and PDF
    
    # Guard against zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for EI calculation
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # --- Combine EI with dynamic exploration ---
    # The final score is a blend of exploitation (EI) and exploration (sigma/cost)
    # Weight the exploration term by the dynamic beta
    
    # Exploration term: high sigma, low cost
    exploration = beta * sigma_safe / cost_safe
    
    # Final score: EI (exploitation) + exploration term
    # Scale EI to be comparable to exploration
    ei_scale = np.mean(ei) if np.mean(ei) > 1e-9 else 1.0
    ei_normalized = ei / ei_scale
    
    # Blend: 70% exploitation, 30% exploration, with dynamic weighting
    score = 0.7 * ei_normalized + 0.3 * exploration
    
    # Additional: mild penalty for very expensive candidates
    cost_penalty = 0.05 * (cost_safe - 1.0)
    score = score - cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
