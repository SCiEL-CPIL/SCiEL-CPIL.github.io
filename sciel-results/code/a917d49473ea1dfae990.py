import numpy as np
from scipy.stats import t as student_t
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before the 330s hard limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    
    Uses Student's t-distribution based Expected Improvement (EI) to account
    for heavy-tailed uncertainty in the limited sample budget (30 evals).
    This promotes more robust exploration of the 6D landscape compared to
    standard Gaussian EI.
    
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
    global _start_time
    
    # Deadline guard: If we are running out of time, return a safe, valid score.
    if time.time() - _start_time > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-8)
        dof = 5.0
        z = (f_best - mu_arr) / sigma_safe
        # For Student-t based EI:
        # E[max(f_best - X, 0)] where X ~ t_dof(mu, sigma)
        # This can be computed using the t-distribution CDF and PDF
        t_cdf = student_t.cdf(z, dof)
        t_pdf = student_t.pdf(z, dof)
        ei = (f_best - mu_arr) * t_cdf + sigma_safe * t_pdf
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Use Student's t-distribution with ~5 degrees of freedom
    # This accounts for heavy-tailed uncertainty in the limited 30-sample budget
    dof = 5.0
    
    # Standardize for t-distribution
    z = (f_best - mu) / sigma_safe
    
    # Compute t-distribution based EI
    # For a t-distributed variable, the expected improvement is:
    # EI = (f_best - mu) * T_dof(z) + sigma * t_dof(z)
    # where T_dof is the CDF and t_dof is the PDF of the t-distribution with dof degrees of freedom
    t_cdf = student_t.cdf(z, dof)
    t_pdf = student_t.pdf(z, dof)
    
    ei = (f_best - mu) * t_cdf + sigma_safe * t_pdf
    
    # Add a small exploration term to balance exploitation and exploration
    exploration_weight = 0.1
    exploration_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
