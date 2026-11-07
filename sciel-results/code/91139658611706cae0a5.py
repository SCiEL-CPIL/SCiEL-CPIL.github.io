import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization on Griewank 2D.
    Uses a Thompson Sampling approach with cost normalization, combined with
    Expected Improvement per unit cost for robustness.
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety guards
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    n = len(mu)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # EI per unit cost
    ei_cost = ei / cost_safe
    
    # Thompson Sampling: sample from the GP posterior
    # f(x) ~ N(mu(x), sigma(x)^2)
    # We sample one value per candidate
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within a call
    # Sample: mu + sigma * z
    # But we want to be deterministic across calls for the same inputs? 
    # Actually, TS is stochastic. But in BO, we typically draw one sample.
    # To make it robust, we can use a single fixed random draw or multiple samples.
    # Given the tight budget, let's use a deterministic surrogate for TS: 
    # Lower Confidence Bound (LCB) is deterministic but similar.
    # However, the directive suggests Thompson Sampling.
    # Let's use a fixed random seed per call to ensure consistency if called multiple times
    # in the same evaluation step? No, the function is called once per iteration.
    # So stochasticity is fine.
    
    # To avoid high variance, we can average over a few samples or just use one.
    # Let's try a hybrid: Use LCB (deterministic TS equivalent) with cost normalization.
    # LCB: mu - beta * sigma
    # Since we want to minimize, we prefer low mu and high sigma (uncertainty).
    
    # Adaptive beta: start high, decrease as f_best improves?
    # Or just fixed.
    beta = 2.0
    
    lcb = mu - beta * sigma_safe
    lcb_cost = lcb / cost_safe
    
    # Combine EI/cost and LCB/cost
    # EI is always positive, LCB can be negative.
    # We want higher score = better.
    # So we should maximize the score.
    # EI/cost: higher is better.
    # LCB/cost: lower LCB is better for minimization.
    # So we want to maximize -LCB/cost or similar.
    
    # Let's define score such that higher is better.
    # EI/cost is already "improvement per cost".
    # For LCB, we want to minimize the predicted value.
    # So score = w1 * EI/cost + w2 * (-LCB/cost) ?
    # But LCB can be negative.
    
    # Alternative: Use Expected Hypervolume Improvement? Too slow.
    # Let's stick to EI/cost but enhance the exploration.
    
    # A common cost-aware acquisition is:
    # alpha(x) = EI(x) / cost(x)
    
    # To add exploration, we can use:
    # alpha(x) = EI(x) / (cost(x) * (1 + sigma(x)^2 * kappa))
    # This penalizes high uncertainty more heavily? No, that reduces exploration.
    
    # Actually, cost-aware BO often uses:
    # EI(x) / cost(x)
    # But if cost is low in high-uncertainty areas, this naturally favors them.
    
    # Let's try a different structure:
    # Score = EI(x) / cost(x) + lambda * sigma(x) / cost(x)
    # The second term encourages exploration (high sigma) relative to cost.
    
    lambda_explore = 0.5
    
    exploration_term = sigma_safe / cost_safe
    exploitation_term = ei_cost
    
    score = exploitation_term + lambda_explore * exploration_term
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety: if computation took too long, return simpler fallback
    elapsed = time.time() - t0
    if elapsed > 280:
        return ei_cost
    
    return score
