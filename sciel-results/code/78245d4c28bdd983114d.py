import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining POI, Normalized EI, and Normalized UCB.
    
    Improvements over previous version:
    1. Normalizes EI and UCB by a robust estimate of the current uncertainty scale (90th percentile of sigma).
       This ensures that the weights for POI, EI, and UCB are comparable regardless of the 
       magnitude of the objective function or the current spread of the GP predictions.
    2. Adjusts weights to slightly favor EI (exploitation) over POI, as EI provides 
       better convergence rates once the GP has some data, while POI remains strong 
       for robustness in multimodal landscapes.
    3. Maintains the cost-aware division to prioritize cheaper evaluations.
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
    
    # 1. Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # 2. Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 3. UCB-style exploration term for minimization
    beta = 2.0
    ucb_linear = (f_best - mu + beta * sigma_safe)
    ucb_linear = np.maximum(ucb_linear, 0.0)
    
    # Determine the scale of uncertainty for normalization
    # Using a high percentile makes it robust to outliers (very high sigma in unexplored regions)
    sigma_ref = np.percentile(sigma_safe, 90)
    if sigma_ref < 1e-8:
        sigma_ref = 1.0
        
    # Normalize EI and UCB to be comparable to POI (which is in [0, 1])
    ei_norm = ei / sigma_ref
    ucb_norm = ucb_linear / sigma_ref
    
    # Weights:
    # POI: 0.3 (Robustness)
    # EI:  0.5 (Convergence/Exploitation)
    # UCB: 0.2 (Exploration)
    w_poi = 0.3
    w_ei = 0.5
    w_ucb = 0.2
    
    # Combine terms, weighted by cost
    scores = (w_poi * poi + w_ei * ei_norm + w_ucb * ucb_norm) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return simple POI/cost
        fallback_scores = poi / cost_safe
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
