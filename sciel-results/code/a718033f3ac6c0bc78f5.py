"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) with a
    Lower Confidence Bound (LCB) component. The blend is adaptive:
    early on we explore more (higher LCB weight), later we exploit more.
    
    Since we don't have access to the number of evaluations, we use the
    relationship between sigma and f_best to infer exploration needs.
    Points with high uncertainty relative to the gap to f_best get more
    exploration credit.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # LCB component: mu - beta * sigma
    # Use a moderate beta that balances exploration
    beta = 1.5
    lcb = -mu + beta * sigma  # We want higher scores = more likely to be good
    
    # Adaptive blending: where EI is small (little expected improvement),
    # give more weight to LCB to encourage exploration
    # Normalize EI to [0, 1] roughly using the max EI as scale
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # LCB normalized: higher LCB means more exploration
    lcb_max = np.max(lcb) if np.max(lcb) > 0 else 1.0
    lcb_min = np.min(lcb)
    lcb_range = lcb_max - lcb_min if lcb_range > 1e-12 else 1.0
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Blend: weight LCB more where EI is low
    # w_ei is high where EI is strong, w_lcb is high where EI is weak
    w_lcb = 0.3 * (1.0 - ei_norm) + 0.1  # base exploration weight
    w_ei = 1.0 - w_lcb
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
