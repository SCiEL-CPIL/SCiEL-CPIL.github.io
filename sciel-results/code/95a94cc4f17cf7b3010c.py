"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Strategy: Thompson Sampling with Cost-Awareness.
    
    Instead of a deterministic acquisition function (like EI, PI, or LCB), we use
    Thompson Sampling. We draw a single sample from the GP posterior for each candidate:
    f_sample(x) = mu(x) + z * sigma(x), where z ~ N(0, 1).
    
    Then, we score each candidate by the "value" of this sample relative to its cost.
    Since we want to minimize f(x) but pay a cost, we want to find points where the
    sampled function value is low *and* the cost is low. A common way to combine these
    is to minimize the ratio f_sample(x) / cost(x) (if f can be negative, this is tricky,
    but for minimization where f_best is positive and we are looking for improvements,
    we can look at the potential gain).
    
    However, a more robust approach for minimization with cost is to consider the
    "expected improvement per unit cost" or simply a weighted sum. 
    Given the directive to use a "reward-to-go ratio" or similar stochastic approach:
    Let's define the score as the negative of the sampled value divided by the cost.
    Score = -f_sample(x) / cost(x)
    
    Why this might work:
    1. **Exploration-Exploitation**: Thompson Sampling naturally balances exploration
       (high sigma leads to high variance in f_sample) and exploitation (low mu leads
       to low f_sample).
    2. **Cost Awareness**: By dividing by cost, we explicitly penalize expensive points.
       A point with f_sample = 1.0 and cost = 1.0 has score -1.0.
       A point with f_sample = 0.5 and cost = 2.0 has score -0.25.
       A point with f_sample = 2.0 and cost = 1.0 has score -2.0.
       We want to maximize the score (so we pick the highest score).
       Wait, if we maximize -f_sample/cost, we are minimizing f_sample/cost.
       This favors low f_sample and low cost.
       
    Let's refine:
    We want to select the candidate that minimizes f(x) subject to cost constraints or
    by balancing them. 
    If we simply maximize -f_sample(x) / cost(x), we are looking for the best "bang for buck".
    
    Implementation details:
    - Draw z from N(0, 1) for each of the 2000 candidates.
    - Compute f_sample = mu + z * sigma.
    - Compute score = -f_sample / cost.
    - Return the scores. The evaluator picks the max score.
    
    Note: We must handle division by zero or very small costs.
    Cost is defined as 1 + mean(x_scaled), so cost >= 1.0.
    So cost is always >= 1.0. No division by zero risk.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive and not too small
    cost = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: Draw a sample from the posterior for each candidate
    # z ~ N(0, 1)
    z = np.random.randn(len(mu))
    
    # Sampled function values
    f_sample = mu + z * sigma
    
    # Score: We want to maximize this.
    # We want low f_sample and low cost.
    # Score = -f_sample / cost
    # This is equivalent to minimizing f_sample / cost.
    
    # Alternative: Use a weighted combination if ratio is unstable?
    # The ratio approach is what the directive suggests ("reward-to-go ratio").
    
    score = -f_sample / cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Fallback to simple exploitation: -mu / cost
         score = -mu / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
