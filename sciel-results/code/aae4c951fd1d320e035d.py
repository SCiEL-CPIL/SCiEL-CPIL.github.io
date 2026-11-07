"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 315.0  # seconds, safety margin before 330s kill
_T0 = time.time()


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard expected improvement for minimization."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, beta=2.0):
    """Upper confidence bound for minimization (lower is better)."""
    return mu - beta * sigma


def _probability_of_improvement(mu, sigma, f_best, xi=0.0):
    """Probability of improvement for minimization."""
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu - xi) / sigma
    pi = norm.cdf(z)
    return np.clip(pi, 0.0, 1.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
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
    
    # Compute multiple acquisition components
    # 1. Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best, xi=0.0)
    
    # 2. UCB with adaptive beta based on uncertainty
    # Use a larger beta when sigma is high relative to the range of mu
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    beta = 1.5 + 2.0 * (np.mean(sigma) / (mu_range + 1e-10))
    beta = np.clip(beta, 1.0, 5.0)
    ucb = _ucb(mu, sigma, beta)
    
    # 3. Probability of Improvement
    pi = _probability_of_improvement(mu, sigma, f_best, xi=0.0)
    
    # 4. Cost-aware normalization: divide by cost to get return per unit cost
    # This directly optimizes marginal return on budget
    score_ei = ei / cost
    score_ucb = -ucb / cost  # Negate because lower UCB is better
    score_pi = pi / cost
    
    # Blend: primarily EI/cost, with UCB/cost for exploration and PI/cost for stability
    # Weights sum to 1.0
    w_ei = 0.55
    w_ucb = 0.30
    w_pi = 0.15
    
    score = w_ei * score_ei + w_ucb * score_ucb + w_pi * score_pi
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Add a small random tiebreaker to avoid always picking the same point
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    tiebreaker = rng.uniform(0, 1e-8, size=len(score))
    score = score + tiebreaker
    
    return score
# EVOLVE-BLOCK-END
