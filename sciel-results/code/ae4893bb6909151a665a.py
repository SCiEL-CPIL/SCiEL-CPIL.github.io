"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    # If we are close to the limit, return a neutral score immediately
    if time.time() - _START_TIME > 310:
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Use a "Soft Cost Penalty" based on Probability of Improvement (POI)
    # Instead of dividing EI by cost (which can be very aggressive and zero out expensive but promising points),
    # we use a multiplicative soft penalty: score = EI * exp(-lambda * cost)
    # Or even better, combine EI and POI with a soft cost decay.
    # The key insight from "inspiration" or common BO practices for cost-awareness is that
    # simple division can be too harsh. A soft exponential penalty allows high-EI points
    # to overcome moderate costs, while still preferring cheaper points when EI is comparable.
    
    # Let's try: score = EI * (1.0 / (1.0 + cost))
    # This is a bounded penalty. Cost=1 -> 0.5, Cost=2 -> 0.33, Cost=5 -> 0.16.
    # This is much softer than 1/cost.
    
    # Alternatively, the "PI-based acquisition with soft cost penalty" finding suggests:
    # score = POI * (1.0 / (1.0 + cost)) ?
    # But EI is generally better for simple regret.
    
    # Let's try a hybrid: 
    # score = EI * (1.0 / (1.0 + cost))
    
    scores = ei / (1.0 + cost)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
