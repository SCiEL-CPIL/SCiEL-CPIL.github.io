import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    This implementation uses a structural change from the previous linear blend:
    It employs Cost-Weighted Expected Improvement (C-EI) combined with 
    Cost-Weighted Uncertainty Reduction.
    
    Score(x) = EI(x) / cost(x) + gamma * sigma(x) / cost(x)
    
    This formulation directly maximizes the expected improvement per unit cost,
    while maintaining exploration in cheap, uncertain regions.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, indicating potential for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness:
    # The cost is provided in the 'cost' array (shape 2000).
    # We divide the acquisition signal by the cost to prioritize cheaper evaluations.
    
    # Primary term: Cost-Weighted Expected Improvement
    # This favors points that offer high improvement relative to their cost.
    cei = ei / cost
    
    # Secondary term: Cost-Weighted Uncertainty
    # This favors points with high uncertainty (sigma) relative to their cost.
    # This helps in early exploration and escaping local minima in the surrogate.
    # Gamma controls the balance. A small value ensures exploitation dominates
    # when EI is strong, but allows exploration when EI is flat.
    gamma = 0.1
    cwu = gamma * (sigma_safe / cost)
    
    # Combine the terms
    scores = cei + cwu
    
    # Ensure finite scores
    # Handle any numerical issues that might arise from extreme values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
