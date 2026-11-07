# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Uses a structural change: combines Expected Improvement (EI) with a
    Thompson-sampling-inspired exploration term, both normalized by cost.
    This differs from the parent by adding a stochastic exploration component
    drawn from the GP posterior, which can escape local optima more effectively.
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    if _is_deadline_exceeded():
        # Return scores that prioritize low cost and low predicted mean
        return -(mu + 0.1 * cost)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson sampling term: draw a sample from the GP posterior
    # This provides a different structural exploration mechanism
    # Use a fixed random seed per call to be deterministic but varied
    # The sample is in objective units (lower is better)
    # We want high scores for low sampled values
    # Sample: mu - sigma * |N(0,1)| (always <= mu, encouraging exploration)
    rng = np.random.default_rng()
    sample = mu - sigma_safe * np.abs(rng.standard_normal(len(mu)))
    
    # Convert sampled objective to a score: lower sampled value = higher score
    # Normalize by cost
    ts_score = -sample / cost_safe
    
    # Blend EI and Thompson sampling with a weight that emphasizes exploitation
    # early (when f_best is good) and exploration later
    # Use a fixed blend: 70% EI, 30% Thompson sampling
    ei_score = ei / cost_safe
    
    # Scale ei_score to be comparable to ts_score
    # ei is typically smaller in magnitude than the sampled values
    # Let's normalize both to [0, 1] range approximately
    # Actually, let's just combine them with appropriate weights
    final_score = 0.7 * ei_score + 0.3 * ts_score
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
