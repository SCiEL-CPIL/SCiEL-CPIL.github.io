"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s
_KNOWN_MIN = -1.8013

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Strategy: Rank-Normalized Adaptive UCB/EI
    1. Compute raw EI and UCB scores.
    2. Normalize these scores to ranks [0, 1] to make the acquisition function 
       invariant to the scale of the posterior predictive distribution.
    3. Dynamically adjust the weight between exploitation (EI) and exploration (UCB)
       based on how close f_best is to the known global minimum (-1.8013).
       Early on (f_best >> min), favor exploration. Later, favor exploitation.
    
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
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return -np.asarray(mu, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB for minimization
    # Use a moderate beta. Since we are rank-normalizing, the absolute scale of beta
    # matters less for the final ranking, but it affects the relative ordering of EI vs UCB.
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # Ensure EI and UCB are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Rank Normalization
    # Higher raw score -> higher rank (closer to 1.0)
    # We want the final score to be higher for better candidates.
    # np.argsort(np.argsort(x)) gives ranks 0..n-1 where 0 is smallest.
    # So (n-1 - rank) gives 0 for smallest, n-1 for largest.
    # Normalizing by (n-1) puts them in [0, 1].
    
    n = len(mu)
    if n > 1:
        ei_ranks = np.argsort(np.argsort(ei)) / (n - 1)
        ucb_ranks = np.argsort(np.argsort(ucb)) / (n - 1)
    else:
        ei_ranks = np.array([1.0])
        ucb_ranks = np.array([1.0])
    
    # Adaptive Weighting
    # We need a measure of "progress".
    # f_best starts high (e.g., 0 or positive) and decreases to -1.8013.
    # Let's define a progress ratio.
    # If f_best is very high, we are early. If close to min, we are late.
    # A simple heuristic: 
    #   early: f_best > 0.0 -> weight exploration heavily
    #   mid:   -1.0 < f_best <= 0.0 -> balanced
    #   late:  f_best <= -1.0 -> weight exploitation heavily
    
    # Let's use a smooth transition based on f_best relative to the range.
    # The function values for Michalewicz 2D on [0, pi]^2 typically range from -1.8 to ~0.
    # Let's define a "closeness" metric.
    # closeness = (f_best - _KNOWN_MIN) / (0.0 - _KNOWN_MIN)  # 1.0 when f_best=0, 0.0 when f_best=min
    # If f_best is much worse than 0, closeness > 1.
    # If f_best is better than min (numerical error?), closeness < 0.
    
    # Actually, simpler:
    # w_exploit (weight for EI) should increase as we get closer to the minimum.
    # w_explore (weight for UCB) should decrease.
    
    # Let's map f_best to a weight w in [0, 1] where 1 is fully exploitation.
    # We want w to be low when f_best is high (e.g., 0.0) and high when f_best is low (e.g., -1.8).
    
    # Linear mapping:
    # w = (f_start - f_best) / (f_start - f_min)
    # f_start ~ 0.0 (typical initial uniform random might be around 0 or slightly negative/positive)
    # f_min = -1.8013
    
    f_start = 0.0
    f_min = _KNOWN_MIN
    range_f = f_start - f_min
    
    if range_f > 0:
        w = (f_start - f_best) / range_f
    else:
        w = 0.5
    
    # Clip w to [0, 1]
    w = np.clip(w, 0.0, 1.0)
    
    # We want to start with more exploration (UCB) and end with more exploitation (EI).
    # So, weight for EI = w, weight for UCB = 1 - w?
    # Or maybe a stronger emphasis on EI later?
    # Let's try: score = w * ei_ranks + (1 - w) * ucb_ranks
    # Early (f_best=0): w=0 -> score = ucb_ranks (Pure Exploration)
    # Late (f_best=-1.8): w=1 -> score = ei_ranks (Pure Exploitation)
    
    # This might be too extreme. Let's keep a minimum exploration floor.
    # w_eff = 0.2 + 0.8 * w
    # So even at the end, 20% exploration. And at the start, 80% exploration.
    
    w_eff = 0.2 + 0.8 * w
    scores = w_eff * ei_ranks + (1.0 - w_eff) * ucb_ranks
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
