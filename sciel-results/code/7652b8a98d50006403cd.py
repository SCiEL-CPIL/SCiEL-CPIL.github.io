"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    Strategy:
    1. The current program uses a Linear UCB (Lower Confidence Bound for minimization)
       Score = (beta * sigma - mu) / cost.
    2. This strategy is Linear in mu and sigma.
    3. We switch to a Non-Linear, Probabilistic strategy: Probability of Improvement (PI).
    4. PI = Phi((f_best - mu) / sigma).
    5. Score = PI / cost.
    
    This is structurally different from UCB (non-linear vs linear) and distinct from 
    the seed's EI (bounded probability vs unbounded expected value).
    PI focuses on the likelihood of finding a better point, rather than the magnitude 
    of the improvement or a linear confidence bound. This can lead to different 
    exploration-exploitation dynamics, particularly in regions with high uncertainty 
    where PI might be high even if the expected improvement is low, or vice versa.
    """
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # We want P(f(x) < f_best) = P(Z < (f_best - mu) / sigma)
    
    # Handle sigma = 0 case
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher is better
    # Divide by cost to penalize expensive evaluations
    scores = pi / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
