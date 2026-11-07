import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic xi: requires a relative improvement
    # Early on (high f_best), xi is large, so we need a big drop to improve.
    # This pushes exploration to regions with high potential drop (high sigma or low mu).
    # Later (low f_best), xi is small, so we exploit.
    xi = 0.1 * f_best
    
    z_xi = (f_best - xi - mu) / sigma_safe
    ei_xi = (f_best - xi - mu) * norm.cdf(z_xi) + sigma_safe * norm.pdf(z_xi)
    
    # Blend standard EI and xi-EI
    # Weight standard EI more as we get closer to the optimum?
    # Or just use ei_xi?
    # ei_xi is 0 if f_best - xi - mu is very negative.
    # Let's use ei_xi as the primary score.
    
    scores = ei_xi
    
    # Add a small LCB component to ensure exploration of high uncertainty areas
    # even if EI is low
    lcb = -(mu + 1.0 * sigma_safe) # Minimize mu + sigma, so maximize -(mu+sigma)
    
    # Normalize? EI and LCB are in different scales.
    # EI is usually smaller than LCB term.
    # Let's just add a small fraction of LCB.
    scores = scores + 0.1 * lcb
    
    return scores
