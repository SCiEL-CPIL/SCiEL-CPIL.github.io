"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson Sampling.
    
    Samples from the posterior for each candidate, then scores by the sampled
    value normalized by evaluation cost. This implicitly balances exploration
    and exploitation without heuristic tuning parameters.
    
    For minimization: we want to minimize the objective. Thompson Sampling
    samples y ~ N(mu, sigma^2) and picks the candidate with the lowest y.
    To make it cost-aware, we score as -y / cost (higher is better).
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # Use a fixed seed per call for determinism? No, we want stochastic exploration.
    # But for reproducibility in testing, we might want to be careful.
    # The task says "2000 new uniform candidates", so each call is independent.
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(size=mu.shape)
    
    # For minimization, we want low values. 
    # Score = -sampled_value / cost (higher score = better)
    # But we need to be careful: if f_best is very low, we want to focus on low mu.
    # Thompson sampling naturally handles this: candidates with low mu will tend
    # to have low samples.
    
    # Actually, for cost-awareness, we want to prefer candidates where the
    # expected improvement per cost is high. With Thompson sampling, we can
    # think of it as: sample y, then pick min(y/cost) or equivalently max(-y/cost).
    # But we should also consider that we want to improve on f_best.
    
    # Let's use a hybrid: Thompson sample, but shift by f_best to focus on improvement
    # Actually, pure Thompson sampling for minimization: pick candidate with min sampled value.
    # Cost-aware: pick candidate with min(sampled_value / cost).
    
    # Score = -sampled_value / cost (so higher is better)
    scores = -samples / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
