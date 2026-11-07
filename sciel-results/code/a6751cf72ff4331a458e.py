import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    Uses a Thompson Sampling approach with a cost-aware ratio.
    Draws a sample from the GP posterior and scores by -sample / cost^0.5.
    This provides a principled exploration/exploitation trade-off while
    softly penalizing high-cost evaluations, allowing exploration of
    high-cost regions where shifted minima may reside.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)

    # Thompson Sampling: draw a single random sample from the GP posterior
    # Using a fixed seed per call would make it deterministic, but we want
    # stochastic exploration. However, for reproducibility in a single run,
    # using the global RNG is fine. The evaluator calls this many times.
    # We use np.random.default_rng() without a seed to get fresh randomness.
    # Note: In a strict production setting, we might want a persistent RNG,
    # but here we just need a sample.
    rng = np.random.default_rng()
    samples = rng.normal(mu, sigma)

    # Cost-aware score: minimize the sampled value per unit cost
    # Using cost^0.5 for a softer penalty than linear cost
    # Higher score = better (lower sampled value / lower cost)
    scores = -samples / np.sqrt(safe_cost)

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
