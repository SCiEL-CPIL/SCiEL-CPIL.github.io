"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level time tracking
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill


def _elapsed():
    return time.time() - _start_time


def _time_exhausted():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and PoI with an
    uncertainty-reduction (GEI-like) penalty.
    
    Structural idea (from inspiration 1): blend EI with PoI and add a soft
    penalty that steers evaluations toward under-explored high-sigma regions,
    approximating a lightweight Greedy Expected Improvement without expensive
    covariance updates.
    
    Refinement machinery (from current program): variance penalty and
    cost-normalization, plus deadline guard.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Deadline guard: if time is running out, return a fast fallback
    if _time_exhausted():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard EI: E[max(f_best - f(x), 0)]
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # PoI: P(f(x) < f_best) = Phi(z)
    poi = cdf_z
    
    # Blend EI and PoI: EI captures magnitude of improvement, PoI is robust
    # to GP miscalibration. Weight PoI by sigma so it contributes more in
    # uncertain regions.
    blended = ei + 0.5 * sigma_safe * poi
    
    # Uncertainty-reduction penalty (GEI approximation):
    # Evaluating a point where sigma is already low provides less information
    # gain. Penalize candidates whose sigma is below the median to encourage
    # exploration of under-explored regions.
    sigma_median = np.median(sigma_safe)
    if sigma_median > 1e-8:
        penalty_raw = np.maximum(0.0, sigma_median - sigma_safe) / sigma_median
    else:
        penalty_raw = np.zeros_like(sigma_safe)
    
    # Scale penalty: lambda controls strength
    lambda_penalty = 0.3
    penalty = lambda_penalty * penalty_raw * sigma_safe
    
    # Cost normalization: divide by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Final score: blended improvement per unit cost, minus uncertainty penalty
    scores = (blended - penalty) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to reasonable range to avoid numerical blowup
    scores = np.clip(scores, -10.0, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
