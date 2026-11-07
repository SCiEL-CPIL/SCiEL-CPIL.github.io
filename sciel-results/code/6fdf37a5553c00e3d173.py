"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement (kept for potential fallback or reference)
    # z = (f_best - mu) / sigma_safe
    
    # Optimistic Improvement (OI) / Linear UCB for minimization
    # OI = (f_best - mu) + beta * sigma
    # This is linear in mu and sigma, unlike EI which is non-linear.
    # It is more robust to GP miscalibration in high-uncertainty regions.
    # The parameter beta controls the trade-off between exploration (sigma) and exploitation (f_best - mu).
    # A higher beta encourages more exploration.
    # Based on the cell finding: "Replacing non-linear EI with a linear UCB-based score (Optimistic Improvement) reduces sensitivity to GP miscalibration... Measured: 1 -> 1.614."
    
    # We need to choose beta. 
    # In standard UCB, beta is often a constant or a function of time/iterations.
    # Since we don't have iteration count directly, but we have cost, we can use a fixed beta.
    # The cost is in [1, 2] roughly (1 + mean(x)).
    # The objective scale is around -3.86 to maybe -1 or 0.
    # A reasonable beta might be in the range of 0.1 to 1.0.
    # Let's try beta = 0.5 as a starting point, which is a common choice for UCB-like methods.
    
    beta = 0.5
    
    # OI score
    oi = (f_best - mu) + beta * sigma_safe
    
    # Cost-awareness:
    # The task specifies cost = 1 + mean(x).
    # We want to penalize high-cost points.
    # The previous successful strategy was EI / cost.
    # Let's apply the same cost normalization to OI.
    # scores = oi / cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple OI/cost
        scores = oi / cost
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    scores = oi / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
