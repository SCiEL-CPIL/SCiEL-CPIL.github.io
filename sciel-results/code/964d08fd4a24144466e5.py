"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) per unit cost.
    Structural change: Shifts from local Expected Improvement (EI) to UCB,
    which explicitly balances exploitation (-mu) and exploration (beta*sigma).
    This helps escape local minima in multimodal functions like Rastrigin.
    
    Parameters:
    -----------
    mu : array, shape (2000,)
        GP mean predictions
    sigma : array, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline check: if we are running out of time, return a fast, simple score
    if (time.time() - _START_TIME) > _DEADLINE_SECONDS:
        # Fallback: Exploitation only (lower mu is better)
        return -np.asarray(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # UCB for minimization:
    # We want to maximize the score.
    # Low mu is good (exploitation).
    # High sigma is good (exploration).
    # Score = -mu + beta * sigma
    
    # Adaptive exploration parameter beta:
    # Start with high exploration, reduce as f_best improves (gets closer to 0).
    # Rastrigin min is 0.
    # If f_best is high (e.g., 50), we are early, use large beta.
    # If f_best is low (e.g., 1.0), we are late, use smaller beta.
    
    # Heuristic: beta scales with current best value, capped.
    # f_best is in objective units.
    # Typical Rastrigin values: 0 to 200.
    # If f_best > 10, beta = 2.0 (strong exploration)
    # If f_best < 1, beta = 0.5 (strong exploitation)
    
    # Smooth transition
    # beta = 0.5 + 1.5 * min(1.0, f_best / 10.0)
    
    # Let's try a fixed beta first, as adaptive might be tricky without knowing the scale of f_best exactly in all phases.
    # However, knowing f_best is 0 at the end, we should exploit heavily.
    # Let's use a simple decay based on f_best.
    
    # Clip f_best to non-negative
    f_best_pos = max(0.0, f_best)
    
    # Beta starts at 2.0 and decays to 0.5 as f_best approaches 0.
    # Using a logarithmic or linear scale?
    # Linear: beta = 0.5 + 1.5 * (f_best / 10.0) clipped to [0.5, 2.0]
    beta = 0.5 + 1.5 * np.clip(f_best_pos / 10.0, 0.0, 1.0)
    
    # Calculate UCB score
    # Note: -mu dominates if mu is large positive.
    # sigma adds exploration.
    scores = (-mu + beta * sigma) / cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
