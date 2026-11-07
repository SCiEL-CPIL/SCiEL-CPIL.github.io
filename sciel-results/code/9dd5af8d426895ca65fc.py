"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_EPS = 1e-9
_START_TIME = None
_DEADLINE = 250.0  # safety margin before 330s kill


def _ensure_deadline():
    """Return True if we should continue, False if we've hit the safety deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < _DEADLINE


def _compute_pi(mu, sigma, f_best):
    """Compute Probability of Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    return np.maximum(pi, 0.0)


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Thompson Sampling with cost-awareness.
    
    We sample from the posterior (mu + sigma * z) and score each candidate
    by the negative sampled value divided by cost. This naturally balances
    exploration and exploitation. The division by cost makes the acquisition
    cost-aware: cheap points are favored.
    
    We use a single random draw per call (deterministic seed based on f_best
    and array hashes for reproducibility within a run, but effectively random
    across calls since f_best changes).
    
    Args:
        mu: numpy array of shape (2000,), predicted means
        sigma: numpy array of shape (2000,), predicted standard deviations
        f_best: float, best observed value so far
        cost: numpy array of shape (2000,), evaluation costs for each candidate
        
    Returns:
        numpy array of shape (2000,), acquisition scores (higher is better)
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Convert to numpy arrays for consistent handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Thompson Sampling: sample from the posterior
    # Use a deterministic but effectively random seed based on current state
    # to ensure we get different samples on each call
    seed = int(abs(hash((f_best, float(np.mean(mu)), float(np.mean(sigma))))) % (2**31))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=mu.shape)
    
    # Sampled objective values (lower is better)
    sampled_values = mu + sigma * z
    
    # Score: negative sampled value divided by cost (higher is better)
    # This favors points that are likely to be good (low sampled value)
    # and cheap to evaluate
    scores = -sampled_values / cost
    
    # Add a small exploration bonus based on uncertainty to prevent
    # premature convergence: points with high sigma get a slight boost
    # This is a structural addition to pure Thompson Sampling to help
    # escape local optima in the acquisition landscape
    # The bonus is proportional to sigma but scaled down to not dominate
    sigma_bonus = 0.1 * sigma / cost
    
    scores = scores + sigma_bonus
    
    # Check deadline
    if not _ensure_deadline():
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
