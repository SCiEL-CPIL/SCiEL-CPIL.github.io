import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a combination of Expected Improvement and a UCB-like exploration term.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well within 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    # Use scipy for accuracy if available, otherwise use numpy
    try:
        from scipy.stats import norm
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
    except ImportError:
        # Fallback: approximate using numpy
        from math import sqrt, exp
        sqrt_2pi = sqrt(2.0 * np.pi)
        z = (f_best - mu) / sigma
        # Approximate CDF using erf approximation
        # Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        # Use a simple polynomial approximation for erf
        def erf_approx(x):
            # Abramowitz and Stegun approximation
            sign = np.sign(x)
            x = np.abs(x)
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            p = 0.3275911
            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
            return sign * y
        
        phi_approx = 0.5 * (1.0 + erf_approx(z / sqrt_2pi))
        pdf_approx = np.exp(-0.5 * z * z) / sqrt_2pi
        ei = (f_best - mu) * phi_approx + sigma * pdf_approx
        ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # Use a moderate exploration coefficient
    beta = 0.5
    exploration = beta * sigma
    
    # Combine EI and exploration
    # Scale EI to be comparable with exploration term
    max_ei = np.max(ei) if np.max(ei) > 0 else 1.0
    score = ei + exploration
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result found so far
        return score
    
    return score
