"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Use a global deadline guard to ensure we return before the 300s limit
_START_TIME = time.time()
DEADLINE = 330.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Check if we are close to the deadline
    if time.time() > DEADLINE:
        # Return a uniform random score to break ties or just zero
        # Actually, we should return something finite. 
        # If we are late, the evaluator might kill us. 
        # But we must return valid scores.
        # Let's just return a simple score quickly.
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement (PI) is more robust in high dimensions
    # than Expected Improvement (EI) because it doesn't over-penalize
    # high variance as much, but we can combine them or use a modified version.
    # The inspiration suggests PI shifts search away from high-variance noise.
    
    # Calculate improvement (without xi for PI, or with small xi)
    # Let's use a small xi to ensure we only explore if we expect improvement
    xi = 0.0
    
    improvement = f_best - mu - xi
    
    # Calculate z-score
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Calculate Probability of Improvement
    pi = norm.cdf(z)
    
    # To avoid pure exploitation, we can add a small exploration term
    # Or use a hybrid: alpha * PI + (1-alpha) * EI
    # Let's stick to PI as per inspiration, but scale it by cost.
    
    # Cost-aware score: higher is better.
    # Cost is in [1, 2]. Dividing by cost favors cheaper evaluations.
    score = pi / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
