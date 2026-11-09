"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Quantile-based Expected Improvement (QEI) targeting 95th percentile
    # to robustly exploit uncertainty in multimodal landscape
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Quantile-based term: encourage exploration where uncertainty is high
    # but with a penalty to avoid pure random exploration
    # Use 95th percentile of improvement distribution
    # P(I > 0) = Phi(z), E[I | I > 0] = (f_best - mu) + sigma * phi(z)/Phi(z)
    # The 95th percentile of improvement is approximately:
    # q_0.95 = (f_best - mu) + sigma * norm.ppf(0.95)
    # But we want to balance exploitation and exploration
    
    # Hybrid: 70% EI + 30% quantile-based exploration
    # Quantile-based: higher score when sigma is large relative to improvement
    q_term = (f_best - mu) + sigma * norm.ppf(0.95)
    
    # Normalize to combine
    scores = 0.7 * ei + 0.3 * q_term
    
    return scores
# EVOLVE-BLOCK-END
