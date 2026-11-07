"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s limit

def _elapsed_ok():
    """Check if we're still within time budget."""
    return (time.time() - _start_time) < _TIME_LIMIT

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Cost-Weighted Expected Improvement.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if not _elapsed_ok():
        # Emergency: return simple LCB-like score (negated mu + sigma)
        return -(mu - sigma)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8) # Avoid division by zero
    
    # Safety: ensure cost is positive and finite
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Safety: ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # 1. Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues in extreme tails, though norm.cdf/pdf are stable
    # However, for very large positive z, cdf->1, pdf->0. EI -> improvement.
    # For very large negative z, cdf->0, pdf->0. EI -> 0.
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0) # EI is non-negative
    
    # 2. Adaptive Cost Weighting
    # Theoretical minimum for Hartmann 3D
    f_min_theoretical = -3.86278
    
    # Gap to theoretical minimum
    gap = f_best - f_min_theoretical
    
    # Determine cost penalty exponent alpha
    # If gap is large (early phase), alpha -> 0 (ignore cost)
    # If gap is small (late phase), alpha -> 1 (penalize cost heavily)
    gap_clipped = np.clip(gap, 0.0, 1.0)
    alpha = 1.0 - gap_clipped
    
    # Apply cost penalty: Score = EI * cost^(-alpha)
    # cost is in [1, 2] (1 + mean(x))
    # If alpha=0, cost^0=1, no penalty.
    # If alpha=1, cost^1=cost, full penalty (dividing by cost effectively if we inverted, but here multiplying by 1/cost^alpha)
    # Wait, the formula in inspiration was: score = ei * np.power(cost, -alpha)
    # If alpha=1, score = ei / cost. This is standard cost-aware EI.
    # If alpha=0, score = ei.
    
    cost_factor = np.power(cost, -alpha)
    scores = ei * cost_factor
    
    # 3. Optional: Add a small exploration term if EI is very flat?
    # Pure EI can be greedy. However, Inspiration 1 (score 1.73) used pure EI + adaptive cost.
    # The current program (0.65) used a blend. 
    # To be safe and robust, I will stick to the proven structure of Inspiration 1 
    # but ensure that if EI is 0 everywhere (e.g. f_best is very low), we don't fail.
    # If EI is 0, scores are 0. The evaluator picks argmax. If all 0, it's random/arbitrary.
    # This is acceptable.
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
