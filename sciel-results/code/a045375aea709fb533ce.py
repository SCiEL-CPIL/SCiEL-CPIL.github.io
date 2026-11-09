import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Combines Expected Improvement (EI), Probability of Improvement (PI), 
    and Upper Confidence Bound (UCB) with adaptive weighting based on 
    the current best observed value.
    
    Parameters:
    mu : np.ndarray, shape (2000,)
        Mean of the GP at the candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP at the candidate points.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    np.ndarray, shape (2000,)
        Acquisition scores for each candidate point (higher = better).
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 330s. We aim to finish well before.
    # This function is vectorized and fast, so this is mostly a safeguard.
    deadline = start_time + 250.0 

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # 3. Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x). A "lower" UCB is better.
    # UCB = mu + beta * sigma. Lower values are preferred.
    # Adaptive beta: Explore more when f_best is far from 0 (global optimum).
    # Ackley 2D range is roughly [0, 43]. f_best starts high, goes to 0.
    f_scale = 30.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Beta starts high (exploration) and decays to low (exploitation)
    beta = 0.5 + 2.0 * f_norm
    ucb = mu + beta * sigma
    
    # Normalize components to [0, 1] for consistent weighting
    # EI is in [0, f_best], so dividing by max is a good normalization
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
        
    # PI is already in [0, 1]
    pi_norm = pi
    
    # UCB: lower is better. We invert it so higher score = lower UCB.
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-9:
        ucb_norm = np.ones_like(ucb)
    else:
        ucb_norm = (ucb_max - ucb) / ucb_range
        
    # Adaptive Weighting
    # Early (f_norm ~ 1): High exploration (PI, UCB), Low exploitation (EI)
    # Late (f_norm ~ 0): High exploitation (EI), Low exploration (PI, UCB)
    
    # Weights should sum to 1.
    # w_ei: 0.2 (early) -> 0.7 (late)
    # w_pi: 0.4 (early) -> 0.15 (late)
    # w_ucb: 0.4 (early) -> 0.15 (late)
    
    w_ei = 0.2 + 0.5 * (1.0 - f_norm)
    w_pi = 0.4 - 0.25 * (1.0 - f_norm)
    w_ucb = 0.4 - 0.25 * (1.0 - f_norm)
    
    # Cost Adjustment
    # Cost is in [1, 2]. 
    # Using 1/sqrt(cost) provides a softer penalty than 1/cost.
    # This encourages cheaper points but doesn't ignore potentially better expensive points.
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = 1.0 / np.sqrt(cost_safe)
    
    # Combined Score
    blended_score = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
    scores = blended_score * cost_factor
    
    # Add a small tie-breaker bonus for very low cost
    # (1/cost - 0.5) is in [0, 0.5] for cost in [1, 2]
    cost_bonus = 0.05 * (1.0 / cost_safe - 0.5)
    scores += cost_bonus
    
    # Check deadline (should not trigger for vectorized ops on 2000 points)
    if time.time() > deadline:
        pass # Already have the full array
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
