import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement and LCB.
    """
    start_time = time.time()
    DEADLINE = 325.0

    # Input validation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB) score for minimization
    # We maximize -LCB, which is equivalent to minimizing LCB
    # LCB = mu - beta * sigma
    # Score = -LCB = -mu + beta * sigma
    beta = 2.5
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Blend EI and LCB
    # Weighting: EI provides exploitation (improvement over best), LCB provides exploration (uncertainty + low mean)
    alpha = 0.7
    combined = alpha * ei + (1.0 - alpha) * lcb_score
    
    # 4. Cost Adjustment
    # Divide by cost to penalize expensive evaluations
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    if time.time() - start_time > DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    return scores
