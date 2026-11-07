import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For Gaussian posterior: EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to explore regions with high uncertainty
    # LCB = mu - kappa * sigma (lower is better for minimization)
    # For scoring (higher is better), we use -LCB = -mu + kappa * sigma
    kappa = 2.0
    exploration_score = -mu + kappa * sigma
    
    # Normalize both scores to zero mean, unit variance across the candidate set
    # This prevents one term from dominating due to scale differences
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    exp_mean = np.mean(exploration_score)
    exp_std = np.std(exploration_score) + 1e-8
    exp_norm = (exploration_score - exp_mean) / exp_std
    
    # Weighting: Favor exploration more heavily given the small budget (30 evals)
    # and multimodality of the Weierstrass function
    # Based on observations, a 0.6/0.4 or 0.7/0.3 split favoring exploration works well
    alpha = 0.6  # Weight for exploration (LCB)
    beta = 1.0 - alpha  # Weight for exploitation (EI)
    
    scores = alpha * exp_norm + beta * ei_norm
    
    # Deadline guard - though this is O(N) and fast, we comply with the directive
    if time.time() - start_time > 290:
        # Fallback to pure exploitation if somehow it took too long
        return -mu
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
