"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time and call count
_T0 = [None]
_CALL_COUNT = [0]


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI with a time-aware exploration bonus.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Best observed function value (scalar)
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Initialize timer on first call
    if _T0[0] is None:
        _T0[0] = time.time()
    
    elapsed = time.time() - _T0[0]
    _CALL_COUNT[0] += 1
    
    # Deadline guard with safety margin (330s limit, use 310s for safety)
    if elapsed > 310.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Fallback: simple exploration-based score
        score = sigma_safe / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Base score: EI per unit cost
    base_score = ei / cost_safe
    
    # Add a small exploration bonus that decays over time
    # This helps in early iterations to explore more, then focuses on exploitation
    # The bonus is scaled by sigma to encourage exploration of uncertain regions
    # and divided by cost to account for evaluation cost
    # Decay factor: start with a moderate bonus, reduce as time/iterations progress
    # We estimate progress from the call count (each call = one acquisition evaluation)
    # But we don't know the exact iteration, so use elapsed time as a proxy
    # Budget is ~30 cost units, with ~15 evaluations after 5 initial
    # Total time budget is 330s, so average ~22s per evaluation
    # Use elapsed time to modulate the exploration bonus
    
    # Exploration bonus: encourages visiting uncertain, cheap regions
    # Use both time and f_best progress to modulate exploration
    time_fraction = min(elapsed / 300.0, 1.0)
    
    # As f_best approaches 0, reduce exploration (we're close to optimum)
    # f_best starts around some value and decreases toward 0
    # Use a soft measure: if f_best < 0.1, we're close
    f_best_factor = max(0.0, 1.0 - f_best / 1.0)  # 1 when f_best=0, 0 when f_best>=1
    
    # Combine time decay and progress-based decay
    # Start with higher exploration, reduce as we progress
    exploration_weight = 0.5 * (1.0 - 0.5 * time_fraction) * (1.0 - 0.5 * f_best_factor)
    
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine base EI score with exploration bonus
    score = base_score + exploration_bonus
    
    # Numerical safety
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
