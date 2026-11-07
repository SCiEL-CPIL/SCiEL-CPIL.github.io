"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a global start time to ensure we finish before the 330s limit.
_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. UCB for exploration
    3. Weighted combination that adapts based on uncertainty
    
    The weight shifts from exploration to exploitation as f_best improves
    relative to the initial range.
    """
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Emergency: return pure EI to be fast and valid
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Upper Confidence Bound (UCB) for exploration
    # Use a beta that balances exploration
    # For Hartmann 3D, we want moderate exploration
    beta = 1.5
    ucb = mu + beta * sigma
    
    # For minimization, we want to find low values
    # EI is positive when mu < f_best (improvement expected)
    # UCB: lower mu + beta*sigma means more promising for minimization
    # Actually for minimization, we should use LCB: mu - beta*sigma
    # But higher score means "evaluate next", so we want high scores for good candidates
    
    # Let's think about this more carefully:
    # We're minimizing, so good candidates have low mu
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    #   This is positive when mu < f_best
    #   
    # For UCB in minimization context, we want to explore where sigma is high
    # and mu is low. A common approach is to use:
    #   score = -mu + beta * sigma  (higher is better for selection)
    # or
    #   score = mu - beta * sigma  (lower is better, but we want higher scores)
    #   
    # Since we return scores where HIGHER means evaluate next:
    #   For minimization, we want high scores for low mu and/or high sigma
    #   ucb_score = -mu + beta * sigma  would work
    
    # Let's use a normalized combination
    # Normalize EI and UCB scores to comparable scales
    
    # EI is already in objective units, positive for improvement
    # For the exploration component, use -mu + beta*sigma
    ucb_score = -mu + beta * sigma
    
    # Adaptive weighting: start with more exploration, shift to exploitation
    # as we get closer to the optimum
    # Estimate progress: if f_best is close to known minimum, exploit more
    # Hartmann 3D minimum is approximately -3.86278
    # Initial random points might give values around -1 to -2
    
    # Use a simple heuristic: weight based on the ratio of EI to max possible improvement
    # Or just use a fixed blend that works well for 30 evaluations
    
    # For 30 evaluations on Hartmann 3D, a good strategy is:
    # - Early: more exploration (UCB weight ~0.5-0.7)
    # - Late: more exploitation (EI weight ~0.7-0.9)
    
    # We can estimate "late" by looking at how tight sigma is overall
    # If sigma is small everywhere, we're in exploitation phase
    
    # Simple approach: use the mean sigma as a proxy for exploration need
    mean_sigma = np.mean(sigma)
    
    # If mean sigma is high, we're still exploring; if low, exploiting
    # Normalize mean_sigma to [0, 1] roughly
    # Hartmann 3D has values in roughly [-4, 0] range
    sigma_norm = np.clip(mean_sigma / 0.5, 0, 1)  # 0.5 is a reasonable max sigma
    
    # Weight: when sigma_norm is high (early), use more UCB
    # When sigma_norm is low (late), use more EI
    w_ucb = 0.3 + 0.4 * sigma_norm  # Range: [0.3, 0.7]
    w_ei = 1.0 - w_ucb
    
    # Normalize both to have similar scale
    # EI range: typically 0 to ~1-2
    # UCB score range: depends on mu and sigma
    
    # Let's normalize by their respective max values
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 1e-9 else 1.0
    
    # Combine
    score = w_ei * (ei / ei_max) + w_ucb * (ucb_score / ucb_max)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
