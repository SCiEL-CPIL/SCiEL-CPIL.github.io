"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as student_t
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Student-t based expected improvement.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Check deadline
    if _check_deadline():
        # Return a safe fallback: just use negative mu (exploitation)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score for standard normal EI
    z = (f_best - mu) / sigma
    
    # Standard EI (Gaussian)
    ei_gauss = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Student-t based EI: use t-distribution with moderate degrees of freedom
    # This accounts for additional model uncertainty
    df = 5.0  # degrees of freedom for Student-t
    
    # For Student-t, we need to compute expected improvement under t-distribution
    # The t-distribution has heavier tails, so we adjust the EI calculation
    # Using the fact that if X ~ t_df, then E[max(f_best - X, 0)] can be approximated
    
    # Normalize: if X ~ N(mu, sigma^2), under t_df approximation,
    # we can scale: Y = (X - mu) / sigma * sqrt(df / (df - 2)) for df > 2
    # But a simpler approach: use the t CDF and PDF directly
    
    # The standardized variable: z_t = (f_best - mu) / (sigma * sqrt(df/(df-2)))
    # Actually, let's use a more direct approach:
    # Under Student-t with df degrees of freedom, if we standardize:
    # T = (X - mu) / (sigma * sqrt(df / (df - 2))) ~ t_df (approximately, for df > 2)
    
    scale_factor = np.sqrt(df / (df - 2.0))  # ~1.118 for df=5
    z_t = (f_best - mu) / (sigma * scale_factor)
    
    # Expected improvement under t-distribution:
    # E[max(f_best - X, 0)] = sigma * scale_factor * [ (1 - F_t(z_t)) * z_t + f_t(z_t) ]
    # where F_t is CDF and f_t is PDF of t_df
    # Note: this is the same form as Gaussian EI but with t distribution functions
    
    t_cdf_val = student_t.cdf(z_t, df)
    t_pdf_val = student_t.pdf(z_t, df)
    
    ei_student = (f_best - mu) * t_cdf_val + sigma * scale_factor * t_pdf_val
    
    # Combine Gaussian and Student-t EI (blend for robustness)
    # Use Student-t as primary, Gaussian as secondary
    ei_combined = 0.7 * ei_student + 0.3 * ei_gauss
    
    # Cost-aware scaling: divide by cost
    # Use a power less than 1 to be less aggressive on cost
    cost_power = 0.8
    scores = ei_combined / (cost ** cost_power)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
