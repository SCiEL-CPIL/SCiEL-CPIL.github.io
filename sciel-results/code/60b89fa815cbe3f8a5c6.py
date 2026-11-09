"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: Normalized Blend of Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    This approach addresses the trade-off between exploitation (EI) and 
    exploration (UCB) by normalizing both scores to a common [0, 1] scale 
    before blending. This prevents one metric from dominating due to scale 
    differences and allows for a robust balance across different stages of 
    optimization.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    # norm.cdf(z) is Phi(z), norm.pdf(z) is phi(z)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # We want low mu and high sigma.
    # Score = -mu + kappa * sigma
    kappa = 2.0
    ucb_scores = -mu + kappa * sigma_safe
    
    # 3. Normalize scores to [0, 1] to make them comparable
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range < 1e-9:
            return np.ones_like(s) * 0.5
        return (s - s_min) / s_range

    ei_norm = normalize_scores(ei_scores)
    ucb_norm = normalize_scores(ucb_scores)
    
    # 4. Blend the normalized scores
    # A 50/50 blend provides a stable compromise.
    # In early stages, UCB_norm will likely be high for unexplored areas.
    # In late stages, EI_norm will likely be high for promising areas.
    # Normalization ensures both contribute meaningfully.
    w_ucb = 0.5
    w_ei = 1.0 - w_ucb
    
    final_scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    return final_scores
# EVOLVE-BLOCK-END
