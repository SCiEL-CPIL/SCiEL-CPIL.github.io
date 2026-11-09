"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB and Expected Improvement.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Deadline guard: if we're getting close to the limit, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Simple fallback: just use -mu (exploitation only)
        return -np.asarray(mu, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # UCB term: encourages exploration of high-uncertainty regions
    # beta is the exploration parameter; larger beta = more exploration
    beta = 2.0
    ucb = -mu + beta * sigma  # negated because we want to minimize
    
    # Expected Improvement term (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both terms to comparable scales
    # Scale EI by its max so it's comparable to UCB
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_scaled = ei / ei_max
    else:
        ei_scaled = ei
    
    ucb_max = np.max(ucb)
    if ucb_max > 1e-12:
        ucb_scaled = ucb / ucb_max
    else:
        ucb_scaled = ucb
    
    # Blend UCB and EI with a weighting that balances exploration and exploitation
    # Weight UCB more heavily to escape local minima in multimodal landscape
    w_ucb = 0.6
    w_ei = 0.4
    
    scores = w_ucb * ucb_scaled + w_ei * ei_scaled
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
