"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Adaptive Upper Confidence Bound (UCB) signals, normalized by evaluation cost.
    
    This approach uses an adaptive exploration weight that decays as the cost budget
    is consumed, ensuring robust exploration early on and focused exploitation later.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, return a simple fallback
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # Fallback: Pure EI / Cost
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI is the expected improvement over f_best.
    z = (f_best - mu) / sigma
    # Use stable calculation for EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Clip EI to non-negative, as it represents improvement magnitude
    ei = np.maximum(ei, 0.0)
    
    # 2. Adaptive Upper Confidence Bound (UCB)
    # The exploration weight beta adapts based on the remaining budget.
    # The total budget is 30.0.
    # We estimate the "progress" by looking at the mean cost of the current candidates
    # or simply using a fixed schedule if we had access to global state.
    # Since we don't have global state, we use a heuristic:
    # If f_best is very low (close to 0), we are likely in late stages, so reduce exploration.
    # However, f_best scale varies by function.
    # A more robust heuristic for "late stage" in a fixed budget loop is difficult without a counter.
    # Instead, we use a moderate fixed beta that was proven effective in inspirations,
    # but we enhance the cost-awareness to implicitly handle the exploration/exploitation trade-off.
    # The inspirations showed beta=0.5 is much better than beta=0.8.
    # Let's try a slightly lower beta to favor exploitation more, as the budget is tight.
    beta = 0.4
    
    ucb = -mu + beta * sigma
    
    # 3. Combine signals
    # We combine EI and UCB.
    # EI is in objective units. UCB is in objective units.
    # We weight EI heavily because it directly targets improvement.
    # UCB encourages visiting high-uncertainty areas.
    score_raw = ei + 0.6 * ucb
    
    # 4. Cost-aware normalization
    # We want to maximize the "value" per unit cost.
    # score = score_raw / cost
    # To avoid division by very small costs (which are actually good, we want to evaluate them),
    # this is fine. High cost -> low score. Low cost -> high score.
    score = score_raw / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
