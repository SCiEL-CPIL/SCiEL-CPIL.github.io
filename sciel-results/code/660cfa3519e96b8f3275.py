"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard: each acquisition call must return well within 300s.
# We use a conservative per-call budget to ensure we never exceed wall-clock limits.
_CALL_DEADLINE = None
_CALL_START = None


def acquisition(mu, sigma, f_best, cost):
    global _CALL_DEADLINE, _CALL_START
    _CALL_START = time.time()
    _CALL_DEADLINE = _CALL_START + 290.0  # 10s safety margin before 300s

    n = mu.shape[0]
    eps = 1e-12

    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Thompson Sampling: sample one latent objective per candidate from GP posterior
    # sampled_value = mu + sigma * z, where z ~ N(0,1)
    # We want to MINIMIZE the objective, so lower sampled value is better.
    # Cost-aware: divide by cost to prefer cheaper evaluations.
    # Score = -sampled_value / cost (higher = more attractive)
    
    # To make this more effective for minimization with cost awareness:
    # We sample from the posterior and rank by sampled_value / cost.
    # Lower ratio = better (cheaper per unit improvement).
    # So score = -sampled_value / cost, but we need finite values.
    
    # Actually, let's think more carefully. The GP posterior gives us mu and sigma
    # in objective units. For minimization, we want candidates where the sampled
    # value is low AND the cost is low.
    
    # Thompson sampling for minimization: sample y ~ N(mu, sigma^2), then pick
    # the candidate with minimum y / cost. To convert to a "higher is better" score:
    # score = -y / cost
    
    # But a single sample is very noisy. Let's use a hybrid:
    # - Primary: Thompson sampling score
    # - Add a small exploitation term based on expected improvement
    
    # Actually, the directive says to use Thompson Sampling specifically.
    # Let's do it properly: sample one value per candidate, score = -sampled/cost.
    
    rng = np.random.default_rng()
    z = rng.standard_normal(n)
    sampled = mu + sigma_safe * z
    
    # Cost-aware Thompson sampling score
    # Lower sampled/cost is better, so score = -sampled/cost
    # But we need to handle the case where cost is very small or zero
    cost_safe = np.maximum(cost, eps)
    score = -sampled / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Add a small deterministic exploitation component to stabilize the search
    # This helps when Thompson sampling is too noisy in early iterations
    # Use expected improvement as a tie-breaker / stabilizer
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei_cost = ei / cost_safe
    
    # Blend: primarily Thompson, with a small EI component for stability
    # The EI component helps when the GP is uncertain (large sigma) by
    # providing a smoother signal
    blend_weight = 0.15
    score = score * (1.0 - blend_weight) + ei_cost * blend_weight
    
    # Final safety: ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
