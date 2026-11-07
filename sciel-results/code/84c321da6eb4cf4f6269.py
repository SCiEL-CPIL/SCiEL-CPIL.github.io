import numpy as np
from scipy.special import erf

def acquisition(mu, sigma, f_best):
       # Ensure sigma is positive and not zero
       sigma_safe = np.maximum(sigma, 1e-9)
       z = (f_best - mu) / sigma_safe
       # PI = Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
       scores = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
       return scores
