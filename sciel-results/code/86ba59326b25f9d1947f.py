"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = 325.0  # seconds (330s limit, 5s safety margin)
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _check_deadline():
    """Return True if we should stop (deadline exceeded)."""
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE


def _ei_score(mu, sigma, f_best):
    """Expected Improvement."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, beta):
    """Lower Confidence Bound for minimization (higher score = more promising)."""
    return -mu + beta * sigma


def _pi_score(mu, sigma, f_best, cost):
    """Probability of Improvement per unit cost."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    return pi / (cost + 1e-9)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: float, lowest observed value
    - cost: array of shape (2000,), evaluation costs
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Check deadline
    if _check_deadline():
        # Return a simple, fast score to avoid timeout
        ei = _ei_score(mu, sigma, f_best)
        return ei / (cost + 1e-9)
    
    n = len(mu)
    
    # Estimate progress: use f_best relative to known optimum
    # Styblinski-Tang 2D minimum is -78.3323
    f_min_known = -78.3323
    
    # If f_best is very close to the minimum, we're in exploitation mode
    # If f_best is far, we're in exploration mode
    # Use a sigmoid-like transition
    # progress = 1 / (1 + exp(-k * (f_best - f_min_known)))
    # When f_best >> f_min_known, progress ~ 0 (exploration)
    # When f_best ~ f_min_known, progress ~ 0.5
    # When f_best << f_min_known (shouldn't happen), progress ~ 1
    
    # Actually, let's think about this differently.
    # Early on, f_best is high (e.g., 0-50), so we want more exploration.
    # Late on, f_best is close to -78.33, so we want more exploitation.
    
    # Use a simple linear interpolation based on f_best
    # f_best starts around 0-50 and ends around -78.33
    # Map f_best from [0, 50] to [0, 1] for progress
    # But f_best could be negative from the start
    
    # Let's use: progress = clip((f_best - f_min_known) / (50 - f_min_known), 0, 1)
    # When f_best = 50, progress = 1 (full exploration)
    # When f_best = -78.33, progress = 0 (full exploitation)
    # Wait, that's backwards. Let me reconsider.
    
    # Actually, for minimization:
    # Early: f_best is high (bad), we want to explore
    # Late: f_best is low (good), we want to exploit
    
    # So progress (toward exploitation) should increase as f_best decreases.
    # progress = 1 - clip((f_best - f_min_known) / (50 - f_min_known), 0, 1)
    # When f_best = 50: progress = 1 - (50 - (-78.33)) / (50 - (-78.33)) = 1 - 1 = 0 (exploration)
    # When f_best = -78.33: progress = 1 - 0 = 1 (exploitation)
    # When f_best = 0: progress = 1 - (0 - (-78.33)) / (50 - (-78.33)) = 1 - 78.33/128.33 = 1 - 0.61 = 0.39
    
    # Hmm, that seems reasonable. But let's use a smoother transition.
    
    # Alternative: use a Gaussian kernel centered at some midpoint
    # Or just use a fixed blend that works well.
    
    # Let's try: 70% EI, 30% UCB with adaptive beta
    # Beta should be larger when sigma is large
    
    # Compute EI score (cost-aware)
    ei = _ei_score(mu, sigma, f_best)
    ei_cost = ei / (cost + 1e-9)
    
    # Compute UCB score with adaptive beta
    # Beta should be larger when uncertainty is high
    # Use beta = 1.5 + 1.0 * (sigma / (sigma + 1.0))
    # This gives beta in [1.5, 2.5]
    beta = 1.5 + 1.0 * (sigma / (sigma + 1.0 + 1e-12))
    ucb = _ucb_score(mu, sigma, beta)
    ucb_cost = ucb / (cost + 1e-9)
    
    # Normalize both to be comparable
    ei_max = np.max(ei_cost) + 1e-12
    ucb_max = np.max(ucb_cost) + 1e-12
    
    ei_norm = ei_cost / ei_max
    ucb_norm = ucb_cost / ucb_max
    
    # Blend: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
