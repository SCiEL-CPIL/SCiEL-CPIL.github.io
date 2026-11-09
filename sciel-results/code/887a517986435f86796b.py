import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with exploration bonus.
    
    Parameters:
    mu (np.array): Mean of GP prediction, shape (2000,)
    sigma (np.array): Std dev of GP prediction, shape (2000,)
    f_best (float): Best observed function value
    cost (np.array): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.array: Acquisition scores, shape (2000,)
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Add exploration bonus
    # Based on lessons, adding an exploration term improves performance.
    # We use a coefficient of 1.0 for the standard deviation.
    exploration_bonus = 1.0 * sigma
    
    # Combine EI and exploration bonus
    # The cost is in the denominator to penalize expensive evaluations.
    numerator = ei + exploration_bonus
    
    # Calculate final score
    # Ensure cost is not zero (it's >= 1 by definition, but safe to check)
    score = numerator / np.maximum(cost, 1e-9)
    
    return score
