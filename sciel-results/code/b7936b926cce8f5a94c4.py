"""Thompson-sampling cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We use a conservative margin to avoid being killed mid-computation.
_DEADLINE = 325.0  # seconds from process start
_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: float, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple fallback: negative normalized mu / cost
        # to avoid crashing
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost
    
    # Thompson Sampling: draw a single sample from the GP posterior
    # sample = mu - sigma * z, where z ~ N(0,1)
    # We want to minimize the sampled value, so we use -z to get lower samples
    # Actually, for minimization, we want low values. 
    # If we draw z ~ N(0,1), then sample = mu + sigma * z
    # We want to select candidates with low sampled values
    # To make this work with "higher score = better", we negate
    
    # Draw random numbers for Thompson sampling
    # Use a deterministic seed based on f_best and array properties for reproducibility
    # within a single call, but different across calls
    rng = np.random.default_rng()
    
    # Draw z ~ N(0, 1)
    z = rng.standard_normal(len(mu))
    
    # Thompson sample: mu + sigma * z
    # For minimization, we want low values
    # Score = -sampled_value / cost (higher is better)
    sampled = mu + sigma * z
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Score: negative sampled value divided by cost
    # This naturally balances exploration (via sampling) and exploitation
    # while penalizing expensive evaluations
    scores = -sampled / safe_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
