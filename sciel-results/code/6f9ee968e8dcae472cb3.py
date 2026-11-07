import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (POI)
    and Expected Improvement (EI), with a linear UCB-style exploration term.
    
    Strategy:
    - POI is robust in low-data regimes for multimodal functions.
    - EI captures magnitude of improvement.
    - A UCB-style term encourages exploration in uncertain regions.
    - All terms are divided by cost to prioritize cheap evaluations.
    
    The combination balances exploitation (finding the minimum) with exploration
    (covering the search space) while being cost-aware.
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 290.0  # 10s safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    n = len(mu)
    
    # Guard against zero/negative sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # 1. Probability of Improvement (POI) - robust for multimodal functions
    poi = norm.cdf(z)
    
    # 2. Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 3. UCB-style exploration term for minimization
    # Encourages exploration when uncertainty is high
    beta = 2.0
    ucb_linear = (f_best - mu + beta * sigma_safe)
    ucb_linear = np.maximum(ucb_linear, 0.0)
    
    # Combine all terms, weighted by cost
    # POI is robust, EI captures magnitude, UCB encourages exploration
    # Weights: POI gets 0.5, EI gets 0.3, UCB gets 0.2
    scores = (0.5 * poi + 0.3 * ei + 0.2 * ucb_linear) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return simple POI/cost if we are extremely close to timeout
        fallback_scores = poi / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
