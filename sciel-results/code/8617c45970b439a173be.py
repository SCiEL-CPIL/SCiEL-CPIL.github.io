"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 300.0  # safety margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-3, neginf=1e-9)
    
    # Clamp sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Deadline guard: if we're running out of time, return a simple score
    if time.time() > _DEADLINE - 5.0:
        return -(mu) + 0.1 * sigma
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score
    # Use a beta that balances exploitation and exploration
    # For Weierstrass, which is highly multimodal, we want more exploration
    beta = 0.5
    ucb = mu + beta * sigma
    
    # Compute a hybrid score:
    # - EI captures exploitation potential
    # - UCB captures exploration
    # - Add a rank-based exploration bonus for high-uncertainty regions
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Normalize UCB (lower is better for minimization, so negate)
    ucb_neg = -ucb
    ucb_range = np.max(ucb_neg) - np.min(ucb_neg)
    if ucb_range > 1e-12:
        ucb_norm = (ucb_neg - np.min(ucb_neg)) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_neg)
    
    # Rank-based exploration bonus: reward points with high sigma that aren't already top-EI
    sigma_rank = np.argsort(np.argsort(sigma))  # rank of sigma, 0 = lowest
    sigma_rank_norm = sigma_rank / max(len(sigma_rank) - 1, 1)
    
    # Weight the components
    # Higher weight on EI for exploitation, moderate on UCB, small on exploration rank
    weight_ei = 0.5
    weight_ucb = 0.3
    weight_explore = 0.2
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm + weight_explore * sigma_rank_norm
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
