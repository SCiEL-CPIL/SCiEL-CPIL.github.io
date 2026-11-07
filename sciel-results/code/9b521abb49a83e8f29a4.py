import numpy as np
import time
from scipy.stats import norm

# Track elapsed time to enforce deadline
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock seconds
_MARGIN = 5.0      # safety margin before limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is a robust choice for Bayesian optimization, especially with a small
    number of evaluations. It naturally balances exploration and exploitation
    based on the current best observed value (f_best) and the posterior
    uncertainty (sigma).
    
    For minimization:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    We return the EI scores. Higher EI means a higher probability of finding
    a value lower than f_best.
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: Pure exploitation (minimize mu)
        # Return -mu so that lower mu gets higher score
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite to avoid numerical issues
    # Replace non-finite sigmas with a small positive value
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, we want to find x such that f(x) < f_best.
    # The standard EI formula assumes we are maximizing f(x) - f_best.
    # Here we minimize f(x), so improvement is f_best - f(x).
    # The distribution of f(x) is N(mu, sigma^2).
    # E[max(f_best - f(x), 0)]
    # Let Y = f(x). E[max(f_best - Y, 0)]
    # If Y ~ N(mu, sigma^2), then f_best - Y ~ N(f_best - mu, sigma^2).
    # Let Z = (f_best - Y - (f_best - mu)) / sigma = (mu - Y) / sigma.
    # This is equivalent to standard EI for maximization of Y' = -Y.
    
    # Using the standard formula for minimization EI:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI is always non-negative.
    # If mu > f_best, z is negative. Phi(z) is small, phi(z) is small.
    # Actually, if mu > f_best, it's unlikely to improve, but possible.
    # The formula correctly handles this.
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a tiny exploration bonus to encourage diversity if EI is 0 everywhere?
    # No, EI naturally becomes 0 where the model is certain f(x) > f_best.
    # This is good behavior.
    
    return scores
