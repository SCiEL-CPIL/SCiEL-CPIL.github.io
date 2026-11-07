"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 320.0  # seconds, safety margin before 330s kill
_T0 = time.time()


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard expected improvement with exploration parameter xi."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, f_best, beta=2.0):
    """Upper confidence bound (lower is better, so we subtract)."""
    return mu - beta * sigma


def _qei(mu, sigma, f_best, q=0.5):
    """Quantile-based EI: expected improvement above the q-quantile of f_best."""
    # For minimization, we want points likely to beat f_best.
    # Use a slightly optimistic f_best to encourage exploration.
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    # Adjust f_best downward to encourage exploration
    f_opt = f_best - 0.01 * sigma.max() if sigma.max() > 0 else f_best
    z = (f_opt - mu) / sigma
    ei = (f_opt - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    - mu: (2000,) mean predictions from GP
    - sigma: (2000,) standard deviation predictions from GP
    - f_best: lowest observed value
    - cost: (2000,) cost of evaluating each point
    
    Returns:
    - (2000,) acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() - _T0 > _DEADLINE:
        # Return best available: highest mu improvement per cost
        return np.zeros(len(mu))
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition strategies and blend
    
    # 1. Expected Improvement per unit cost
    ei = _expected_improvement(mu, sigma, f_best, xi=0.0)
    score_ei = ei / cost
    
    # 2. UCB per unit cost (lower mu is better, so we use -mu + beta*sigma)
    ucb = _ucb(mu, sigma, f_best, beta=2.0)
    # For minimization, lower ucb is better. We want high acquisition score.
    # Normalize: score = (min_ucb - ucb) / cost to make higher = better
    min_ucb = np.min(ucb)
    score_ucb = (min_ucb - ucb) / cost
    
    # 3. Quantile EI per unit cost
    qei = _qei(mu, sigma, f_best, q=0.5)
    score_qei = qei / cost
    
    # 4. Pure exploitation: lowest predicted value per cost
    # Lower mu is better, so score = (min_mu - mu) / cost
    min_mu = np.min(mu)
    score_exploit = (min_mu - mu) / cost
    
    # 5. Exploration: high uncertainty per cost
    score_explore = sigma / cost
    
    # Blend strategies with weights that favor exploitation early, exploration later
    # We don't know the iteration count, but we can use the spread of sigma as a proxy
    sigma_max = sigma.max()
    sigma_min = sigma.min()
    sigma_range = sigma_max - sigma_min + 1e-9
    
    # When uncertainty is high relative to range, explore more
    explore_weight = 0.15
    exploit_weight = 0.35
    ei_weight = 0.30
    ucb_weight = 0.15
    qei_weight = 0.05
    
    score = (ei_weight * score_ei + 
             exploit_weight * score_exploit + 
             explore_weight * score_explore +
             ucb_weight * score_ucb +
             qei_weight * score_qei)
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Add a small random tiebreaker to avoid always picking the same point
    # This helps with diversity in the search
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    tiebreaker = rng.uniform(0, 1e-6, size=len(score))
    score = score + tiebreaker
    
    return score
# EVOLVE-BLOCK-END
