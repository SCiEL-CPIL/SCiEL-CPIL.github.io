"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a pure Lower Confidence Bound (LCB) strategy with a high exploration
    constant to aggressively target regions with high posterior variance.
    
    The LCB for minimization is defined as:
        LCB(x) = mu(x) - k * sigma(x)
    
    We want to find the point with the lowest possible function value.
    The "optimistic" lower bound is mu - k*sigma.
    To select the point with the highest potential for a low value (i.e., the
    lowest lower bound), we minimize (mu - k*sigma).
    
    However, the interface requires returning scores where HIGHER is better.
    Therefore, we return:
        score = -(mu - k*sigma) = -mu + k*sigma
    
    This is structurally identical to UCB for maximization, but interpreted
    here as the negation of the LCB for minimization.
    
    Key Design Choices:
    1. Pure LCB/UCB: Abandon Expected Improvement (EI). EI is known to be
       overly conservative and prone to local exploitation, which is detrimental
       for multimodal functions like Michalewicz 2D where the global minimum
       is isolated.
    2. High Exploration Constant (k=3.0): Michalewicz 2D has sharp, narrow
       minima surrounded by high-gradient ridges. A high k ensures that the
       acquisition function strongly prefers unexplored regions (high sigma)
       over regions where the mean is merely "okay". This helps jump over
       local minima and find the global basin.
    3. Deterministic: No stochastic sampling, ensuring consistent and fast
       evaluation.
    4. No Adaptive Blending: Simple, robust, and avoids the pitfalls of
       tuning complex blending weights that may not generalize well.
    """
    # Ensure sigma is non-negative and avoid division by zero issues if any
    # (though we don't divide by sigma in LCB, it's good practice)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration constant
    # k=3.0 is chosen to be aggressive. 
    # In 2D BO, k is often in the range [1, 5]. 
    # Higher k favors exploration. Given the multimodal nature and the need
    # to find the global minimum among many local ones, we lean towards
    # high exploration.
    k = 3.0
    
    # Compute score: -mu + k * sigma
    # Higher score means more likely to be evaluated next.
    # This corresponds to minimizing (mu - k * sigma).
    score = -mu + k * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
