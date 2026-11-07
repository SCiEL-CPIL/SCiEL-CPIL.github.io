import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI)
    # We want to maximize the acquisition function.
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2), this has a closed form.
    
    # Handle case where sigma is very small to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # z = (f_best - mu) / sigma
    # Note: Since we are minimizing, improvement is when f(x) < f_best.
    # Let Y = f_best - f(x). We want E[max(Y, 0)].
    # f(x) ~ N(mu, sigma^2) => -f(x) ~ N(-mu, sigma^2)
    # Y = f_best - f(x) ~ N(f_best - mu, sigma^2)
    # Let Z ~ N(0, 1). Y = (f_best - mu) + sigma * Z
    # E[max(Y, 0)] = (f_best - mu) * P(Y > 0) + sigma * E[Z | Z > - (f_best - mu)/sigma]
    # P(Y > 0) = P(Z > - (f_best - mu)/sigma) = P(Z < (f_best - mu)/sigma) = Phi(z)
    # where z = (f_best - mu) / sigma
    # The second term is sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
