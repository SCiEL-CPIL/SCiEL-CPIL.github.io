# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid approach.
    
    Strategy: 
    Instead of pure EI, we use a weighted combination of:
    1. Cost-Adjusted Expected Improvement (EI/cost)
    2. Cost-Adjusted UCB (Upper Confidence Bound)
    
    UCB is defined as: mu + kappa * sigma
    For minimization, we want low values, so we look for regions with low mu 
    but potentially high uncertainty (sigma) that could hide a minimum.
    However, standard UCB for minimization is often formulated as minimizing:
    mu - kappa * sigma (Lower Confidence Bound, LCB)
    
    Wait, standard UCB is for maximization. For minimization, we want to 
    minimize the "worst-case" or "likely" value. 
    A common exploration strategy for minimization is to look for low mu 
    and high sigma. 
    Let's use: Score = - (mu - beta * sigma) / cost
    This effectively maximizes (beta * sigma - mu) / cost.
    
    We combine this with EI to retain the exploitation power of EI while 
    adding the structured exploration of UCB/LCB.
    
    We also apply a dynamic beta that might increase early on. Since we don't 
    know the iteration count directly, we can estimate it from the number of 
    points or just use a fixed reasonable beta. Given the short budget (cost < 30),
    we are likely in the early-to-mid phase. A moderate beta (e.g., 1.0-2.0) 
    is usually good.
    
    Let's try a simple blend:
    Score = alpha * (EI / cost) + (1 - alpha) * (LCB / cost)
    
    Actually, EI is already a strong metric. The issue with EI is it can get 
    stuck in local optima if the GP model is confident (low sigma) but wrong 
    (due to multimodality). UCB/LCB encourages visiting high uncertainty areas.
    
    Let's use a normalized score to combine them.
    """
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x). We estimate the potential minimum as mu - beta * sigma.
    # A lower value here is better.
    # So we want to MAXIMIZE the score. 
    # Score_LCB = -(mu - beta * sigma) = beta * sigma - mu
    # Let's pick beta = 1.5 (a common heuristic for balanced exploration)
    beta = 1.5
    lcb_score = beta * sigma_safe - mu
    
    # Normalize EI and LCB scores to be on a similar scale before combining?
    # EI is in objective units. LCB is in objective units.
    # However, EI is always >= 0. LCB can be negative.
    # Let's shift LCB so it's comparable or just weight them.
    
    # Alternative: Use the raw values but ensure they are scaled.
    # If f_best is large, EI is large. If mu is small, LCB is large (less negative).
    
    # Let's try a simple convex combination.
    # We want high EI and high LCB score (which means low mu or high sigma).
    
    # To make them comparable, let's normalize them by their respective scales if possible,
    # or just use a fixed weight.
    
    # A robust approach: 
    # Score = w1 * (EI / cost) + w2 * (LCB_score / cost)
    
    # Let's set w1 = 0.5, w2 = 0.5
    # This effectively creates a "Cost-Aware Hybrid"
    
    scores_ei = ei / cost
    scores_lcb = lcb_score / cost
    
    # Ensure LCB score doesn't dominate negatively if mu is very high
    # Actually, if mu is very high, lcb_score is very negative, which is bad. Good.
    # If mu is very low, lcb_score is high. Good.
    
    # Combine
    scores = 0.5 * scores_ei + 0.5 * scores_lcb
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
