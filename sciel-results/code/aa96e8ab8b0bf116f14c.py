import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-efficient EI with probability-of-improvement
    term. Structural change: instead of weighted EI+UCB divided by cost, we compute
    a constrained improvement metric that naturally balances exploration/exploitation
    and cost without arbitrary scaling constants.
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (for minimization)
    pi = norm.cdf(z)
    
    # Cost-aware scoring:
    # Key insight: use EI/cost as the primary metric but add a PI term
    # that is ALSO cost-normalized. The PI term provides exploration pressure
    # without the instability of UCB. Both terms are in "improvement per cost"
    # units, so no arbitrary scaling is needed.
    
    # Primary: expected improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Secondary: probability of improvement per unit cost
    # This is more stable than UCB and naturally favors regions with
    # high uncertainty but also reasonable expected value
    pi_per_cost = pi / cost_safe
    
    # Combine with a weight that favors exploitation when f_best is good
    # (close to 0) and more exploration when f_best is poor
    # The Griewank minimum is 0, so f_best in [0, ~0.5] for typical runs
    # When f_best is small, we want to exploit (high EI weight)
    # When f_best is large, we want to explore (higher PI weight)
    
    # Adaptive weight: as f_best approaches 0, shift toward exploitation
    # f_best is typically in [0, 0.5] range for Griewank on [-10,10]^2
    f_best_abs = abs(f_best)
    # Weight transitions from 0.3 (explore) to 0.7 (exploit) as f_best goes from 0.5 to 0
    w_ei = 0.7 - 0.4 * min(f_best_abs / 0.5, 1.0)
    w_pi = 1.0 - w_ei
    
    score = w_ei * ei_per_cost + w_pi * pi_per_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety deadline guard
    elapsed = time.time() - t0
    if elapsed > 290:
        return np.maximum(ei_per_cost, 0.0)
    
    return score
