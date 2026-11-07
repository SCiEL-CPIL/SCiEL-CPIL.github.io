"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    This approach uses a pure confidence-bound strategy, distinct from
    improvement-based (EI/PI) or sampling-based (TS) methods.
    It balances exploitation (low mean) and exploration (high uncertainty)
    via the LCB, and normalizes by cost to favor cheaper evaluations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter beta.
    # When f_best is far from the minimum (0), we want to explore more (higher beta).
    # When f_best is close to 0, we want to exploit more (lower beta).
    # Using a logarithmic or square root scaling helps with large f_best values.
    # f_best is positive (Rosenbrock min is 0).
    f_best_val = max(f_best, 1e-6)
    
    # Beta starts at 1.0 and increases slowly as f_best increases.
    # log1p is stable for small values.
    beta = 1.0 + 0.5 * np.sqrt(np.log1p(f_best_val))
    
    # Lower Confidence Bound for minimization:
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize -LCB.
    lcb = mu - beta * sigma
    
    # Cost-aware score:
    # Divide by cost^alpha to penalize expensive points.
    # Alpha = 1.0 provides a linear penalty.
    alpha = 1.0
    cost_penalty = np.power(cost, alpha)
    
    scores = -lcb / cost_penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
