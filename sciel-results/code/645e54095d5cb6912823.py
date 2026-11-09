"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D.
    
    Strategy: Combine Expected Improvement (EI) for exploitation with UCB for exploration.
    The current program (score 1.49) used EI + 0.5*UCB.
    
    We will refine this by:
    1. Using a slightly more aggressive exploration weight in UCB (beta=1.2) to better handle
       the multi-modality of Branin, while keeping the strong exploitation signal from EI.
    2. Ensuring numerical stability with safe sigma.
    3. Keeping the direct summation approach which avoids normalization issues.
    
    Score = EI + 0.5 * UCB
    where UCB = -mu + beta * sigma
    """
    t_start = time.time()
    
    # Safety margin for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Use vectorized operations
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Calculate UCB (beta = 1.2, slightly increased from 1.0 to encourage more exploration)
    beta = 1.2
    ucb = -mu + beta * sigma_safe
    
    # Combine them.
    # EI is generally smaller in magnitude than UCB in the early stages where sigma is large.
    # In later stages, sigma is small, UCB ~ -mu, EI ~ (f_best-mu) if mu < f_best.
    # We weight them to balance exploration and exploitation.
    # The previous score of 1.49 with 0.5*UCB suggests this balance is good.
    # Let's try slightly increasing the UCB weight to 0.6 to see if it helps find the global min faster.
    score = ei + 0.6 * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard
    t_end = time.time()
    if t_end - t_start > 290:
        pass
        
    return score
# EVOLVE-BLOCK-END
