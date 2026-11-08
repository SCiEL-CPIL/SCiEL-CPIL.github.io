import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # If sigma > 0:
    #   EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Else:
    #   EI = max(f_best - mu, 0)
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-8)
    
    # Compute z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_clipped
    
    # Compute EI
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2 / 2) / sqrt(2 * pi)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_clipped * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # cost is the cost of evaluating at x
    scores = ei / (cost + 1e-9)
    
    # Handle any NaN or inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
