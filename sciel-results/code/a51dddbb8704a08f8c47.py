import time
import numpy as np
from scipy.stats import norm

def _norm_cdf(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -10, 10)
    return norm.cdf(z)

def _norm_pdf(z):
    z = np.asarray(z, dtype=np.float64)
    return norm.pdf(z)

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)

    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    cdf_z = _norm_cdf(z)
    pdf_z = _norm_pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Softer cost penalty: 1 / sqrt(cost)
    score = ei / np.sqrt(cost)
    
    score = np.where(np.isfinite(score), score, 0.0)
    
    if time.time() > deadline:
        pass
        
    return score
