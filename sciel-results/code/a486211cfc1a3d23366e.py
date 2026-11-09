"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-Adjusted Lower Confidence Bound (LCB)
    # LCB for minimization is mu - beta * sigma
    # We want to maximize the acquisition, so we use -LCB = -mu + beta * sigma
    # We adjust the exploration term by 1/sqrt(cost)
    # This penalizes exploration in high-cost regions
    
    # Beta controls the exploration-exploitation trade-off
    # A higher beta encourages more exploration
    beta = 2.5
    
    # The exploration term is scaled by inverse square root of cost
    # This makes exploration more attractive in low-cost regions
    exploration_term = beta * sigma_safe / np.sqrt(cost_safe)
    
    # The score is the negative mean plus the adjusted exploration term
    # Higher scores indicate more promising candidates
    scores = -mu + exploration_term
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace -inf with a very small number to ensure return type is float
    # However, the problem says "Return 2000 finite scores". 
    # So we must replace -inf with a finite value, e.g., a very negative number
    # or 0 if we want to be safe, but usually -inf is mapped to low priority.
    # Let's use a large negative number based on the scale of mu.
    # Or simply clip to a reasonable range.
    
    # If any scores are -inf, replace them with a very low value
    # This effectively deprioritizes them
    min_score = np.min(scores[scores > -np.inf]) if np.any(scores > -np.inf) else -1e6
    scores = np.where(np.isfinite(scores), scores, min_score - 1e6)
    
    return scores
# EVOLVE-BLOCK-END
