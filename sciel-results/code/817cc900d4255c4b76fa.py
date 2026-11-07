"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Stochastic Expected Improvement (Thompson Sampling variant).
    
    Instead of computing the deterministic Expected Improvement (EI), we draw a sample
    from the posterior distribution of the objective function at each candidate point.
    The score is the realized improvement of that sample over the current best,
    divided by the evaluation cost.
    
    This approach naturally balances exploration and exploitation:
    - High sigma points have a higher chance of sampling a very low value (optimistic bias).
    - High cost points are penalized by the division.
    - It avoids the "zero gradient" issues of deterministic EI near the optimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct and inputs are safe
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: fall back to deterministic EI if time is running out
    if time.time() - _START > _LIMIT:
        # Deterministic EI fallback
        z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # Generate random samples from the GP posterior: y ~ N(mu, sigma)
    # We use a fixed seed for reproducibility within a single call? No, the evaluator
    # runs multiple iterations. We should use the global random state or a local one.
    # Using np.random.randn is standard.
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Calculate realized improvement: max(0, f_best - sample)
    # Since we are minimizing, improvement is positive if sample < f_best
    improvement = np.maximum(0.0, f_best - samples)
    
    # Cost-aware score: Improvement / Cost
    scores = improvement / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
