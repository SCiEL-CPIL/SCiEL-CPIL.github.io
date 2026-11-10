import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) signals, normalized by evaluation cost.
    
    This approach balances exploitation (finding points likely to improve
    on the current best) with exploration (visiting uncertain regions)
    while respecting the non-uniform cost structure of the problem.
    
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
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB)
    # Budget-aware beta: shift from exploration to exploitation as budget depletes.
    # Use the mean cost as a proxy for remaining budget fraction.
    # When costs are high (late in budget), reduce exploration.
    mean_cost = np.mean(cost)
    # Normalize: typical cost range is [1, 2], so mean_cost in [1, 2]
    # Map to a budget fraction: 1.0 -> early (high exploration), 2.0 -> late (low exploration)
    budget_fraction = np.clip((2.0 - mean_cost) / 1.0, 0.0, 1.0)
    # Beta decays as budget depletes: from 0.8 (early) to 0.2 (late)
    beta = 0.2 + 0.6 * budget_fraction
    
    # For minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # 3. Combine signals with smooth cost-aware weighting
    # Use a softer cost penalty: divide by (1 + alpha * (cost - 1))
    # This avoids extreme penalties for slightly higher cost candidates
    alpha = 0.5  # Cost sensitivity parameter
    cost_penalty = 1.0 + alpha * (cost - 1.0)
    cost_penalty = np.maximum(cost_penalty, 1e-6)
    
    score_raw = ei + 0.5 * ucb
    score = score_raw / cost_penalty
    
    # 4. Probability-weighted boost for high-confidence improvements
    # When P(improvement) is high, boost the score slightly
    p_improve = norm.cdf(z)
    # Smooth boost: add a small term proportional to p_improve * ei
    boost = 0.1 * p_improve * ei
    score = score + boost / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
