import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE = 325.0  # Safety margin before 330s limit

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and an exploration term (sigma) to balance exploitation and exploration.
    
    Args:
        mu: Array of GP mean predictions, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: Scalar, best observed value so far.
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        Array of acquisition scores, shape (N,). Higher is better.
    """
    _ensure_start()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Deadline guard
    if _time_remaining() < 0.5:
        # Fallback to simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = np.clip((f_best - mu) / sigma_safe, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: Use sigma to encourage exploration of uncertain regions.
    # This acts as a diversity/crowding penalty implicitly, as high sigma
    # indicates unexplored areas.
    # We weight it to be comparable to EI.
    # Since Levy values can be large, we might need to scale this.
    # However, sigma is in objective units.
    # Let's try a weight of 1.0 for now, or maybe scale by mean sigma?
    # The cell finding suggested a weight of 0.3 for a UCB-like term.
    # Let's use a weight of 0.5 for the sigma term.
    
    exploration_weight = 0.5
    exploration_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    combined = ei + exploration_term
    
    # Cost-aware score
    scores = combined / cost_safe
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
