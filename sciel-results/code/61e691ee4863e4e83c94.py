"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a stage-adaptive hybrid of Lower Confidence Bound (LCB) and 
    Expected Improvement (EI), normalized by cost.
    
    - Early stage (high f_best): Aggressive LCB to encourage exploration.
    - Late stage (low f_best): Robust EI for precise exploitation.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid overflow in exp for very negative/positive z
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-10)
    
    # EI per unit cost
    score_ei = ei / cost_safe
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the acquisition score, so we use -LCB = beta * sigma - mu
    # Beta controls exploration. 2.0 is a standard choice.
    beta = 2.0
    lcb_score = beta * sigma_safe - mu
    
    # LCB per unit cost
    score_lcb = lcb_score / cost_safe
    
    # Determine stage based on f_best
    # If f_best is large, we are in early exploration (use LCB).
    # If f_best is small, we are in late exploitation (use EI).
    # Transition threshold: 50.0
    # w_ei is the weight for EI.
    # w_ei = 1 when f_best <= 0 (Late)
    # w_ei = 0 when f_best >= 50 (Early)
    # Linear interpolation: w_ei = 1 - f_best / 50
    w_ei = 1.0 - f_best / 50.0
    w_ei = np.clip(w_ei, 0.0, 1.0)
    
    # w_lcb is the weight for LCB
    w_lcb = 1.0 - w_ei
    
    # Combine scores
    scores = w_ei * score_ei + w_lcb * score_lcb
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simple fallback
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback to simple EI/cost
        simple_scores = score_ei
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
