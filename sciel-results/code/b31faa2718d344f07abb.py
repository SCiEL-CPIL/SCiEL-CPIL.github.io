import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Quantile of Improvement (QI)
    # QI = E[f(x) | f(x) < f_best]
    # We want to minimize QI.
    # Score = -QI / cost
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Calculate QI
    # QI = mu - sigma * phi(z) / Phi(z)
    # Handle z where Phi(z) is very small (unlikely improvement)
    # If Phi(z) is close to 0, QI approaches f_best (from below? No, from above? 
    # Actually, if z is very negative, the left tail is very small. The mean of the left tail is close to -infinity? 
    # No, for a normal distribution, the mean of the truncated left tail is mu - sigma * phi(z)/Phi(z).
    # As z -> -inf, phi(z)/Phi(z) ~ -z.
    # So QI ~ mu - sigma * (-z) = mu + sigma * z = mu + f_best - mu = f_best.
    # So QI is bounded by f_best from below? No, QI < f_best always.
    # Wait, if z is negative, mu > f_best.
    # The region f < f_best is the left tail.
    # The mean of the left tail is less than mu.
    # Is it less than f_best? Yes, because it's the mean of values < f_best.
    # So QI < f_best.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Avoid division by zero
    Phi_z_safe = np.maximum(Phi_z, 1e-15)
    
    q_i = mu - sigma_safe * (phi_z / Phi_z_safe)
    
    # We want to minimize q_i.
    # So score = -q_i / cost
    score = -q_i / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
