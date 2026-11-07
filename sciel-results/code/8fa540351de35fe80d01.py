"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = None


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling cost-aware acquisition function.
    
    Instead of deterministic Expected Improvement, we sample a single latent
    function value for each candidate from the GP predictive distribution:
        sample = mu + sigma * z,  where z ~ N(0, 1)
    
    Then we score each candidate by (sample - f_best) / cost, which stochasticizes
    the exploration-exploitation trade-off. This is a structural change from EI:
    it naturally balances exploration (high sigma leads to large samples) with
    exploitation (high mu leads to high samples) and cost awareness (dividing by cost).
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, lowest observed function value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Deadline guard: if we're close to the 300s limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple cost-aware EI
        improvement = f_best - mu
        z = improvement / sigma_safe
        from scipy.stats import norm
        ei = np.maximum(improvement * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Thompson Sampling: draw one standard normal per candidate
    z = np.random.standard_normal(len(mu))
    
    # Sampled latent function values
    samples = mu + sigma_safe * z
    
    # Score: (sample - f_best) / cost
    # We want to maximize this, so higher scores are better
    # Since we're minimizing, a sample much below f_best is great
    # Actually, we want to find low values, so we should score by (f_best - sample) / cost
    # But Thompson sampling typically maximizes the sample value for maximization
    # For minimization, we minimize, so we want low samples
    # The acquisition should be high when we expect a low value
    # So score = (f_best - sample) / cost makes sense: high when sample << f_best
    
    # But wait: Thompson sampling for minimization: sample f(x), pick x with smallest sample
    # To convert to a "higher is better" score: score = -sample / cost or (f_best - sample) / cost
    # Let's use (f_best - sample) / cost, which is positive when sample < f_best
    
    raw_score = (f_best - samples) / cost_safe
    
    # Ensure finite
    raw_score = np.where(np.isfinite(raw_score), raw_score, 0.0)
    
    # Add a small exploration bonus based on uncertainty to encourage exploration
    # Scale by relative sigma
    max_sigma = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    exploration_bonus = 0.1 * (sigma_safe / max_sigma) / cost_safe
    
    score = raw_score + exploration_bonus
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
