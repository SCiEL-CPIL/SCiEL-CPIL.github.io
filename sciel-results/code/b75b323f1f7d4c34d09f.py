import numpy as np
from scipy.stats import t as student_t

def acquisition(mu, sigma, f_best, cost):
    """
    Student-t Cost-aware Expected Improvement (cEI) acquisition function.
    
    Parameters:
    mu (np.ndarray): Mean predictions of shape (2000,)
    sigma (np.ndarray): Standard deviation predictions of shape (2000,)
    f_best (float): Best observed value so far
    cost (np.ndarray): Cost of evaluation at each point of shape (2000,)
    
    Returns:
    np.ndarray: Scores of shape (2000,), higher is better
    """
    # Ensure sigma is positive to avoid division by zero
    sig = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sig
    # Clip z to avoid numerical issues in t-distribution pdf/cdf
    # t-distribution pdf/cdf are well-behaved for moderate z, but extreme z can cause under/overflow
    z = np.clip(z, -50.0, 50.0)
    
    # Degrees of freedom for Student-t distribution
    # Lower nu = heavier tails = more exploration
    # Nu=3 is a common choice for robustness
    nu = 3.0
    
    # Student-t CDF and PDF
    # scipy.stats.t takes df as the first argument
    t_cdf = student_t.cdf(z, nu)
    t_pdf = student_t.pdf(z, nu)
    
    # Expected Improvement for Student-t distribution
    # EI = (f_best - mu) * T_nu(z) + sigma * ((nu + z^2) / (nu - 1)) * t_nu(z)
    # This formula is valid for nu > 1
    factor = (nu + z**2) / (nu - 1.0)
    ei = (f_best - mu) * t_cdf + sig * factor * t_pdf
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring with alpha = 0.8
    # cost^0.8 penalizes cost less than cost^1.0, allowing exploration of higher-cost regions
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / (cost_safe ** 0.8)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
