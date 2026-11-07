import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid approach: Expected Improvement (EI) with a small exploration
    parameter, combined with a UCB-like boost for high-uncertainty regions.
    
    This structural change moves from pure UCB (which can be overly exploratory
    in high dimensions with few samples) to a balanced EI + UCB strategy.
    EI focuses on exploitation (finding low values), while the UCB boost
    ensures sufficient exploration of uncertain regions.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    # Ensure sigma is strictly positive to avoid division by zero in z-score
    # 1e-6 is a safe lower bound given the noise level of 1e-6 in the GP
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Small exploration parameter xi to balance exploitation and exploration
    # A small positive xi encourages exploring points that are slightly worse than f_best
    # but have high uncertainty, preventing premature convergence.
    xi = 0.01
    
    # Calculate the z-score for the expected improvement
    # z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # Phi is the standard normal CDF, phi is the standard normal PDF
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a UCB-like exploration term to encourage exploring high-uncertainty regions
    # This helps escape local minima in the 6D space.
    # The coefficient 0.1 is small enough that EI dominates in low-uncertainty regions
    # but large enough to influence the choice in high-uncertainty regions.
    ucb_boost = 0.1 * sigma_safe
    
    # Combine EI and UCB boost
    scores = ei + ucb_boost
    
    # Ensure all returned scores are finite to satisfy the task contract
    # Replace non-finite values with 0.0 (or a very small negative number)
    # 0.0 is safe as EI is non-negative and UCB boost is non-negative, 
    # so valid scores should generally be > 0. However, if numerical issues occur,
    # we must return finite values.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
