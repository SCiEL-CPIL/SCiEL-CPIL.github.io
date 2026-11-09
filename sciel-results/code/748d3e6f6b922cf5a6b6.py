"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition: sample one posterior draw per candidate,
    return negative sampled value divided by cost (higher = evaluate next).
    
    mu: (2000,) GP mean in objective units
    sigma: (2000,) GP std in objective units
    f_best: lowest observed value
    cost: (2000,) evaluation cost per candidate
    Returns: (2000,) scores, higher = evaluate next
    """
    start = time.time()
    deadline = start + 290.0  # safety margin before 300s limit

    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # This gives a single posterior sample that captures both mean and uncertainty
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(mu.shape)

    # We want to minimize, so lower sampled value is better.
    # Score = -sampled_value / cost (higher score = prefer to evaluate)
    # Add a small offset to handle negative values properly
    scores = -samples / (cost + 1e-9)

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    elapsed = time.time() - start
    if elapsed > 280.0:
        # If we're running long (shouldn't be for this simple computation), return what we have
        return scores

    return scores
# EVOLVE-BLOCK-END
