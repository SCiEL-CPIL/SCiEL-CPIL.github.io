import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) exploration, penalized by cost.
    
    Args:
        mu: Mean of GP predictions (shape 2000,)
        sigma: Standard deviation of GP predictions (shape 2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Acquisition scores for each candidate (shape 2000,)
    """
    # Ensure inputs are numpy arrays for performance
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) exploration term
    # kappa controls the exploration/exploitation balance
    kappa = 2.5
    ucb_term = kappa * sigma_safe
    
    # Combine EI and UCB
    # Using sum to balance exploitation (EI) and exploration (UCB)
    raw_score = ei + ucb_term
    
    # Non-linear cost penalty
    # The measured knowledge suggests an "offset 0.6" non-linear penalty.
    # Dividing by (cost + offset) creates a hyperbolic penalty that is strong
    # for low costs and saturates for high costs, preventing overly conservative
    # behavior compared to linear division or subtraction.
    cost_offset = 0.6
    penalty_denominator = cost + cost_offset
    
    # Final score
    scores = raw_score / penalty_denominator
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
