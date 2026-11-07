import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best + 1e6) # If mu is inf, treat as very bad
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to prevent numerical issues in exp/erf
    # For z > 5, Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu
    # For z < -5, Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate standard normal CDF and PDF
    # Using scipy.stats.norm for accuracy and speed
    Phi_z = norm.cdf(z_clipped)
    phi_z = norm.pdf(z_clipped)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If z is very negative, (f_best - mu) is negative, Phi is 0, so term is 0.
    # If z is very positive, (f_best - mu) is positive, Phi is 1, so term is f_best - mu.
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Exploration Parameter
    # We want to explore more when f_best is high (far from optimum)
    # and less when f_best is low (close to optimum)
    # f_best for Beale can be up to ~160. Optimum is 0.
    # Let's use a decay based on f_best.
    # beta starts at 2.0 and decays to 0.2
    beta_min = 0.2
    beta_max = 2.0
    # scale factor: when f_best is 1, frac is 0.5?
    # f_best / (f_best + 1.0)
    # If f_best = 0, frac = 0 -> beta = beta_min
    # If f_best = 10, frac = 10/11 ~ 0.9 -> beta ~ beta_max
    frac = f_best / (f_best + 1.0)
    beta = beta_min + (beta_max - beta_min) * frac
    
    # Combine EI and Exploration
    # Score = EI + beta * sigma
    # This ensures that even if EI is 0 (no expected improvement),
    # high uncertainty (sigma) gets a boost.
    score = ei + beta * sigma
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
