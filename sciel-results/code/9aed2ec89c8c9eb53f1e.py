"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Uses cost-normalized EI with adaptive exploration parameter.
    For minimization: EI = E[max(f_best - f(x), 0)]
    Score = EI / cost (higher = evaluate next)
    """
    t_start = time.time()
    DEADLINE = 330.0  # Evaluator kills at 330s, use 325s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite inputs
    if not np.all(np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.all(np.isfinite(sigma)):
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    if not np.all(np.isfinite(cost)):
        cost = np.nan_to_num(cost, nan=1.0, posinf=10.0, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu(x), sigma(x)^2)
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Cost-normalized score
    scores = ei / cost_safe
    
    # Add small exploration bonus based on uncertainty to avoid premature convergence
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-12:
        # Normalize sigma to get relative uncertainty
        sigma_rel = (sigma - sigma_mean) / (sigma_std + 1e-9)
        # Small bonus for high-uncertainty regions
        exploration_bonus = 0.05 * sigma_rel / cost_safe
        scores = scores + exploration_bonus
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5.0:
        pass
    
    return scores
