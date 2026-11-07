import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) for each candidate.
       EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
       This captures the expected magnitude of improvement.
    
    2. Normalize by the evaluation cost.
       The cost function is given as an array `cost` with shape (2000,).
       The task states cost = 1 + mean(x_scaled). 
       We divide EI by cost to get "Expected Improvement per unit cost".
       This encourages the optimizer to prefer points in the cheaper half of the domain 
       (where x_scaled is small, i.e., x is closer to -10 in both dimensions) 
       unless the expected improvement in the expensive region is significantly larger.
       
    3. Add a small exploration bonus based on uncertainty relative to cost?
       Actually, standard Cost-EI is often sufficient. However, to ensure we don't 
       get stuck in a local optimum in the cheap region if the GP is uncertain,
       the EI term naturally includes sigma. 
       
       Let's refine: 
       Score = EI / cost
       
       Why this is different from PoI:
       PoI ignores the magnitude of improvement. If we have a 0.001 chance of improving by 100 
       vs a 0.9 chance of improving by 0.001, PoI favors the second (0.9 > 0.001), but EI favors 
       the first (100*0.001 = 0.1 vs 0.001*0.9 = 0.0009). 
       In Griewank, improvements are often small steps. However, the cost asymmetry is key.
       By dividing by cost, we explicitly model the budget constraint.
       
       Let's also consider that `cost` is provided. 
       The task says: "Evaluating x costs 1 + mean(x scaled to [0, 1])".
       The `cost` array passed to acquisition likely contains these costs for the 2000 candidates.
       
       Implementation details:
       - Handle sigma=0 cases.
       - Ensure numerical stability.
       - Return finite scores.
    """
    start_time = time.time()
    
    # Safety check for inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-scores
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability in norm.cdf and norm.pdf
    # norm.cdf is stable for large |z|, but let's be safe.
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If z is very negative, EI is ~0. If z is very positive, EI is ~ f_best - mu.
    
    # Using scipy.stats.norm
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    # Unclipped mu and sigma for the linear term? 
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) is standard.
    # If we clipped z, we should probably use the clipped z for the pdf/cdf parts.
    # The term (f_best - mu) is exact.
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    # We want to maximize EI / cost
    # cost is given. Ensure it's positive.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Handle any NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Optional: Add a tiny bit of exploration if scores are all zero?
    # Not necessary if EI is working correctly.
    
    # Check time
    # The computation is O(N) and vectorized, so it should be very fast (< 1ms).
    # No need for complex deadline guards for this simple vectorized op, 
    # but we keep the structure for compliance.
    
    return scores
