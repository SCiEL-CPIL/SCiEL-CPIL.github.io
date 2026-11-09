import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline Guard:
    # If we have exceeded a safety margin, fall back to exploitation.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu
    
    # Multi-Objective Expected Improvement (MOEI)
    # We want to balance:
    # 1. Exploitation: Standard Expected Improvement (EI)
    # 2. Exploration: A term that rewards high uncertainty (sigma) but is damped when we are already far from f_best
    
    # Standard EI for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: In some formulations, it's (mu - f_best) for maximization.
    # For minimization:
    # Improvement = max(0, f_best - y)
    # E[Improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma
    
    # Numerical stability for very large |z|
    # For large positive z (mu << f_best), EI ~ f_best - mu
    # For large negative z (mu >> f_best), EI ~ 0
    
    # Use scipy.stats.norm for Phi and phi
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)
    
    # Exploration term:
    # We want to explore regions with high sigma, but only if there's a chance of improvement.
    # A common heuristic is to add a term proportional to sigma, but scaled by how "promising" the region is.
    # Let's use a hybrid:
    # Score = EI + beta * sigma * (1 - Phi(z))  ? No, (1-Phi(z)) is small when mu is low.
    # If mu is low (z is large positive), Phi(z) ~ 1, so (1-Phi(z)) ~ 0. Exploration term vanishes. Good.
    # If mu is high (z is large negative), Phi(z) ~ 0, so (1-Phi(z)) ~ 1. Exploration term is beta * sigma.
    # But EI is also ~ 0. So we rely on exploration.
    
    # Let's try a simple weighted sum:
    # Score = EI + lambda * sigma
    # However, pure EI is often good. Adding sigma encourages exploration.
    # Let's use an adaptive lambda or a fixed one.
    # The inspiration suggests "minimizing expected posterior variance" as a second objective.
    # This is akin to maximizing Information Gain or similar.
    # A simple proxy for information gain in GP is related to sigma^2.
    
    # Let's combine EI and a UCB-like term with a fixed beta.
    # Beta should be small enough not to override strong exploitation signals.
    # Beta = 0.5? 1.0?
    # Let's try beta = 0.5.
    
    beta = 0.5
    exploration_term = beta * sigma
    
    scores = EI + exploration_term
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
