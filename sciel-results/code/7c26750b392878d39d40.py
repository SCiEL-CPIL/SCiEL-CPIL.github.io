import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    # Use Expected Improvement (EI) instead of Probability of Improvement (PI)
    # EI provides a stronger gradient towards the minimum by weighting
    # the magnitude of the potential improvement, whereas PI saturates.
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    # Numerical safety
    ei = np.maximum(ei, 0.0)
    return ei / cost
