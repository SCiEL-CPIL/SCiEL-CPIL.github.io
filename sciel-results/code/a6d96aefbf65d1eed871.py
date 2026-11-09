import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement (CAEI)
    # This approximates the value function in a stochastic shortest path problem
    # where the immediate reward is the improvement over f_best,
    # and the transition cost is the evaluation cost.
    
    # Avoid division by zero or negative costs (though cost should be positive)
    cost = np.maximum(cost, 1e-8)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Vectorized calculation
    z = (f_best - mu) / sigma
    
    # Handle potential numerical issues in z
    # If z is very large positive, Phi(z) -> 1, phi(z) -> 0, EI -> f_best - mu
    # If z is very large negative, Phi(z) -> 0, phi(z) -> 0, EI -> 0
    
    # Use stable calculation if possible, but standard scipy is usually fine for typical ranges.
    # To be safe against overflow in exp for phi, we can use the property that for large |z|, terms vanish.
    
    # Standard formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical noise might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # The acquisition score is EI divided by cost
    # This prioritizes points that offer high expected improvement per unit cost.
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
