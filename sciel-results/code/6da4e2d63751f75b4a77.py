"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for timing
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s limit

def _check_deadline():
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Cost-aware Thompson Sampling.
    
    Instead of maximizing Expected Improvement (EI), we sample from the
    posterior distribution of the objective function and divide by cost.
    This introduces exploration: even if a point has slightly lower EI,
    it might get selected if the sampled value is surprisingly good.
    
    We use a Gaussian approximation:
    Sample ~ N(mu, sigma)
    Score = Sample / Cost
    
    To avoid issues with very high costs or very low sigmas, we add small epsilons.
    """
    if _check_deadline():
        # Fallback to simple EI if we are running out of time, though this is 
        # likely to be called quickly.
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Sample from the posterior: f(x) ~ N(mu, sigma)
    # We want to minimize f, so lower is better.
    # In Thompson Sampling for minimization, we typically sample and pick the minimum.
    # Here we return a score where higher = evaluate next.
    # So we should return something that is high when the sampled value is low.
    # Score = -Sample / Cost? Or 1 / (Cost * (Sample + offset))?
    
    # Let's stick to a robust transformation.
    # If we sample s ~ N(mu, sigma), we want to prefer low s.
    # A common trick is to use -s. But we also need to account for cost.
    # Let's use: Score = -s / cost_safe
    # However, if s is very large (bad), score is very negative.
    # If s is very small (good), score is large positive.
    
    # To prevent extreme values from dominating due to noise, we can clip or use a softer metric.
    # But standard TS uses the sample directly.
    # Let's generate the sample.
    rng = np.random.default_rng()
    samples = rng.normal(mu, sigma_safe)
    
    # The objective is minimization. 
    # We want to maximize the score.
    # If we pick the point with the lowest sampled value, we are doing TS.
    # So score should be high when sampled value is low.
    # score = -samples / cost_safe
    
    # One issue: if cost varies wildly, dividing by cost can be unstable.
    # The cost is 1 + mean(x_scaled). Range [1, 2].
    # So 1/cost is in [0.5, 1]. This is a mild penalty.
    
    scores = -samples / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
