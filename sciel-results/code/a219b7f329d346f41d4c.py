"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 300  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions
    - f_best: scalar, lowest observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros(len(mu))
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Expected Hypervolume Improvement approximation using UCB-like term
    # For minimization, we want to explore where uncertainty is high
    # Use a normalized UCB term that balances with EI
    
    # Adaptive exploration weight based on how close we are to the optimum
    # When f_best is large (early), explore more; when small, exploit
    f_best_abs = abs(f_best)
    
    # UCB for minimization: -mu + beta * sigma (we want low mu and high sigma)
    # Normalize UCB to be comparable to EI
    # The scale of EI depends on f_best, so we need to be careful
    
    # Use a simple approach: blend EI and a normalized exploration term
    # Exploration term: sigma / (sigma + f_best_abs) * some_scale
    # This gives high exploration when sigma is large relative to f_best
    
    # Actually, let's use a more principled approach:
    # PI (Probability of Improvement) = Phi(z)
    # This is bounded in [0, 1] and emphasizes exploitation
    
    # Cost-aware scoring: divide by cost to penalize expensive evaluations
    # cost is in [1, 2], so dividing by cost gives a factor of 0.5 to 1.0
    
    # Combine EI with a small exploration bonus
    # The exploration bonus should be small to not override the EI signal
    # Use: score = EI / cost + lambda * sigma / (sigma + 1) / cost
    # where lambda adapts based on f_best
    
    # When f_best is large (far from optimum), we should explore more
    # When f_best is small (close to optimum), we should exploit more
    
    # Adaptive lambda: starts high, decreases as f_best improves
    # For ackley, f_best starts around 1-2 and goes to 0
    lambda_explore = 0.1 * np.exp(-f_best_abs)
    
    # Exploration term: normalized sigma
    explore_term = sigma / (sigma + 1.0)
    
    # Base score: EI / cost
    base_score = ei / cost
    
    # Add exploration bonus
    explore_score = lambda_explore * explore_term / cost
    
    # Combine
    score = base_score + explore_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
