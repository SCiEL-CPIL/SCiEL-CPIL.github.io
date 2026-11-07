import numpy as np
from scipy.stats import norm
import time

_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() > _start_time + _deadline - _safety_margin:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware score
    # Cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample from the posterior for each candidate
    # For minimization, we want to find low values
    # Sample y ~ N(mu, sigma) for each candidate
    # We want to maximize the probability of improvement, which is equivalent to minimizing y
    # So the acquisition score is -y / cost (higher is better)
    
    # Use a single random seed based on the current state for reproducibility within a run
    # But actually, Thompson sampling typically uses fresh random numbers each time
    # To make it deterministic for the same inputs, we could use a hash of the inputs,
    # but that's complex. Instead, we'll use the standard approach of drawing random samples.
    # However, since the function is called multiple times, we want the randomness to be
    # consistent within a call but different across calls to encourage exploration.
    
    # Actually, for Thompson sampling in BO, we draw a sample from the posterior and evaluate.
    # The "acquisition" here is just the sampled value.
    # Since we want to minimize, the score should be -sampled_value / cost.
    
    # Draw random samples
    # Note: Using np.random.normal directly may not be ideal if we need reproducibility,
    # but for Thompson sampling, the randomness is part of the algorithm.
    # The task says "seeded instances", but that's for the function f_k, not for the acquisition.
    # The acquisition itself can be stochastic.
    
    # However, to avoid issues with non-determinism affecting the score comparison,
    # and since the previous program was deterministic, let's consider if a deterministic
    # approximation is better. But the directive says "Model the acquisition as a Thompson
    # Sampling score by sampling a single objective value from the Gaussian posterior".
    
    # Let's draw the samples.
    # We need to be careful about the scale.
    
    # Standard Thompson Sampling for minimization:
    # Sample f_hat(x) ~ N(mu(x), sigma(x))
    # Select x that minimizes f_hat(x) / cost(x) ? Or f_hat(x) * cost(x)?
    # Usually, cost-aware TS minimizes f_hat(x) * cost(x) or f_hat(x) / cost(x) depending on interpretation.
    # If cost is the price to pay, we want to minimize expected cost per unit improvement?
    # The current EI/Cost suggests we want high EI per unit cost.
    # For TS, if we sample y, we want to minimize y, but penalize high cost.
    # A common formulation is to minimize y * cost or y + lambda * cost.
    # Given the previous EI/Cost, let's try -y / cost to maintain the "per unit cost" intuition.
    # Higher score means better. So if y is low (good) and cost is low (good), -y/cost is high.
    
    # Let's use a fixed seed for the random number generator to ensure reproducibility?
    # No, that would make every call identical if the inputs don't change the seed.
    # The inputs (mu, sigma, f_best, cost) change, but if we use a global RNG, it's stochastic.
    # This is standard for TS.
    
    # However, if the evaluator runs the same function multiple times with the same inputs,
    # it will get different results. This is expected for TS.
    
    # Let's implement it.
    samples = np.random.normal(mu, sigma_safe)
    
    # Score: -sample / cost
    # We want to minimize the objective, so lower sample is better.
    # Dividing by cost penalizes expensive points.
    # Negating makes it a maximization problem (higher score = better).
    scores = -samples / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
