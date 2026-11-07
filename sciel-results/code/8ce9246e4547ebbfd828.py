import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Strategy: Cost-Weighted Expected Improvement (CW-EI).
    
    Expected Improvement (EI) balances exploration and exploitation by considering 
    the expected magnitude of improvement over the current best observation. 
    Unlike Probability of Improvement (PI), which only considers the likelihood 
    of improvement, EI rewards candidates with both high probability and high 
    potential gain.
    
    The cost-aware aspect is introduced by dividing the EI by the evaluation cost. 
    This encourages the algorithm to select points that offer high expected 
    improvement per unit cost. Given that the cost is higher near the known 
    minimum region of the Rosenbrock function (depending on the specific scaling 
    and cost definition), this helps avoid wasting expensive evaluations on 
    regions with low expected gain, while still allowing exploration in cheaper 
    regions if the GP predicts significant uncertainty or potential for improvement.
    
    Formula:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score(x) = EI(x) / cost(x)
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0) # Min cost is 1.0
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If z is very large, Phi(z) ~ 1, phi(z) ~ 0, EI ~ f_best - mu
    # If z is very small/negative, EI is small.
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-weighted score
    score = ei / cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation based on cost-adjusted mean
        score = -mu / cost
        
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
