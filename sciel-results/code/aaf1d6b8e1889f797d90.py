import numpy as np
from scipy.stats import norm
import time

# Module-level state to track evaluation count and deadline
_state = {
    'eval_count': 0,
    'start_time': None,
    'deadline': None
}


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: GP-UCB with time-varying beta + normalized EI.
    
    Uses a theoretically grounded exploration parameter beta_t that grows
    with the number of evaluations, combined with Expected Improvement for
    exploitation pressure.
    
    Parameters:
    mu: np.ndarray, shape (N,)
        Predicted mean of the objective function.
    sigma: np.ndarray, shape (N,)
        Predicted standard deviation of the objective function.
    f_best: float
        The best (lowest) objective value observed so far.
        
    Returns:
    np.ndarray, shape (N,)
        Acquisition scores. Higher scores indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Initialize state on first call
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
        _state['deadline'] = _state['start_time'] + 320.0  # 10s safety margin
    
    # Check deadline - if we're running out of time, return conservative scores
    elapsed = time.time() - _state['start_time']
    if elapsed > 300:
        # Emergency: return scores that favor points with lowest mu
        scores = -mu.copy()
        scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
        return scores
    
    # Increment evaluation count
    _state['eval_count'] += 1
    t = _state['eval_count']
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Time-varying exploration parameter for GP-UCB
    # beta_t = sqrt(2 * t * log(1 + t))
    # This provides theoretically grounded exploration that scales with evaluations
    beta_t = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # GP-UCB score: mu - beta_t * sigma (for minimization, we want low mu and high sigma)
    # Actually for minimization, UCB gives: mu + beta * sigma (we want to explore where sigma is high)
    # But we're maximizing the acquisition score, so:
    # ucb_score = -mu + beta_t * sigma (higher is better: low mu and high sigma)
    ucb_score = -mu + beta_t * sigma_safe
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be comparable in scale to UCB
    # Use the range of EI to normalize
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-10:
        ei_normalized = (ei - ei_min) / ei_range
    else:
        ei_normalized = np.zeros_like(ei)
    
    # Normalize UCB to be comparable
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_normalized = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_normalized = np.zeros_like(ucb_score)
    
    # Combine: 70% UCB (exploration) + 30% EI (exploitation)
    # The weights can be tuned, but this provides a balanced approach
    w_ucb = 0.7
    w_ei = 0.3
    
    scores = w_ucb * ucb_normalized + w_ei * ei_normalized
    
    # Add a small bonus for points with very low mu (strong exploitation signal)
    # This helps when the GP has found a promising region
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range > 1e-10:
        mu_bonus = (mu_max - mu) / mu_range * 0.1  # Small bonus for low mu
    else:
        mu_bonus = np.zeros_like(mu)
    
    scores = scores + mu_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
