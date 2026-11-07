"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_T0 = None
_DEADLINE = 325.0


def _get_elapsed():
    """Return elapsed time since program start, initializing _T0 if needed."""
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Uses a Quantile of Improvement (QI) acquisition function that maximizes
    the conditional expected value of the objective given that it improves
    upon the current best. This focuses search on high-probability, high-reward
    regions rather than just the probability of improvement or raw expected gain.
    
    QI = E[f | f < f_best] * P(f < f_best) / P(f < f_best)
       = E[f | f < f_best]
    
    For a Gaussian N(mu, sigma), the conditional expectation given f < f_best is:
    E[f | f < f_best] = mu - sigma * phi(z) / Phi(z) where z = (f_best - mu) / sigma
    
    We combine this with cost-awareness and a small exploration term.
    """
    # Deadline guard
    elapsed = _get_elapsed()
    if elapsed > _DEADLINE:
        # Fallback: return simple -mu / cost
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        cost = np.where(np.isfinite(cost), cost, 1.0)
        cost = np.maximum(cost, 1e-8)
        score = -mu / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -20, 20)
    
    # Compute Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Avoid division by zero in conditional expectation
    Phi_z_safe = np.maximum(Phi_z, 1e-15)
    
    # Conditional expected value given improvement:
    # E[f | f < f_best] = mu - sigma * phi(z) / Phi(z)
    # We want to MINIMIZE this, so we maximize -E[f | f < f_best]
    cond_mean = mu - sigma * phi_z / Phi_z_safe
    
    # The probability of improvement
    prob_improve = Phi_z
    
    # Quantile of Improvement: we use the conditional mean weighted by 
    # a factor that encourages both high probability and high magnitude improvement
    # QI score = -cond_mean * (1 + alpha * log(prob_improve + eps))
    # This balances: low conditional mean (good) and high probability of improvement
    
    # Actually, let's use a cleaner formulation:
    # The expected improvement is EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # The conditional expected improvement given improvement is:
    # E[f_best - f | f < f_best] = (f_best - mu) - sigma * phi(z) / Phi(z)
    # = (f_best - mu) * Phi(z) / Phi(z) - sigma * phi(z) / Phi(z)
    # = [EI - sigma * phi(z) * (1 - Phi(z))/Phi(z)] / Phi(z)  ... this gets complex
    
    # Simpler approach: maximize the conditional expected value of (f_best - f)
    # given that f < f_best, which is:
    # E[f_best - f | f < f_best] = (f_best - mu) - sigma * phi(z) / Phi(z)
    # This is always >= 0 when z > 0 (i.e., mu < f_best)
    
    cond_ei = np.maximum((f_best - mu) - sigma * phi_z / Phi_z_safe, 0.0)
    
    # The probability of improvement acts as a confidence weight
    # We use a power of prob_improve to balance exploration vs exploitation
    # Higher power = more conservative (only pick very likely improvements)
    # Lower power = more exploratory
    
    # Adaptive power based on f_best:
    # When f_best is large (early), be more exploratory (lower power)
    # When f_best is small (late), be more exploitative (higher power)
    if f_best > 50.0:
        power = 0.5
    elif f_best > 10.0:
        power = 0.7
    elif f_best > 2.0:
        power = 0.9
    else:
        power = 1.2
    
    # Score = cond_ei * prob_improve^power / cost
    # This gives high scores to points that are likely to improve AND have
    # high expected improvement magnitude, penalized by cost
    
    score = cond_ei * (prob_improve ** power) / cost
    
    # Add a small UCB-like exploration term to help escape local minima
    # Scale it down as f_best approaches 0
    if f_best > 1.0:
        beta = 0.5 * np.exp(-f_best / 20.0)
    else:
        beta = 0.1
    
    # Exploration term: high uncertainty in low-prediction regions
    # Use sigma * exp(-mu^2 / (2 * scale^2)) to focus on uncertain areas
    # that are not already predicted to be bad
    scale = max(np.std(mu), 1e-8)
    exploration = sigma * np.exp(-((mu - np.min(mu)) ** 2) / (2 * scale ** 2))
    exploration = exploration / (np.max(exploration) + 1e-10)  # Normalize to [0, 1]
    
    score = score + beta * exploration / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if _get_elapsed() > _DEADLINE:
        return np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
