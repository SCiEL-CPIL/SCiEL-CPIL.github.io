"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates using a Thompson Sampling variant.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: simple regret based score
        cost_safe = np.maximum(cost, 1e-9)
        return -(mu - f_best) / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in sigma for Z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling Strategy:
    # 1. Determine the scale of uncertainty in the current candidate set.
    # Using the max sigma provides a robust measure of the "frontier" uncertainty.
    max_sigma = np.max(sigma_safe)
    
    # 2. Sample a target value T from a Gaussian distribution centered at f_best.
    # The standard deviation of this distribution is scaled by the max uncertainty.
    # This allows the algorithm to stochastically decide how much to explore.
    # If T is sampled significantly lower than f_best, it forces exploration.
    # If T is close to f_best, it favors exploitation of low mu points.
    # We use a fixed seed or random? The task says "seeded instances" for the function, 
    # but the acquisition function is called repeatedly. 
    # Standard Thompson Sampling uses a fresh sample each call.
    # We use np.random.default_rng() for better random state management if needed, 
    # but standard np.random is fine and faster for this scalar sample.
    offset_std = max_sigma
    # Ensure offset_std is not zero to avoid degenerate sampling
    if offset_std < 1e-9:
        offset_std = 1e-9
        
    T = np.random.normal(f_best, offset_std)
    
    # 3. Calculate the probability that each candidate has a value lower than T.
    # P(f(x) < T) = Phi((T - mu) / sigma)
    # where Phi is the CDF of the standard normal distribution.
    z = (T - mu) / sigma_safe
    prob_beat_target = norm.cdf(z)
    
    # 4. Cost-aware scoring:
    # We want to maximize the probability of finding a better point per unit cost.
    score = prob_beat_target / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
