"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a single sample from the GP posterior for each candidate and
    divides by the evaluation cost to prioritize cheap, potentially good points.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value (unused in TS core, kept for signature)
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        # Use a deterministic fallback to be safe
        return np.ones_like(mu)

    # Ensure sigma is non-negative for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a single random sample from the posterior distribution N(mu, sigma^2)
    # This is the core of Thompson Sampling: we pick the point that looks best
    # in a random sample of the function.
    eps = np.random.normal(size=mu.shape)
    samples = mu + sigma_safe * eps
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations.
    # This is a standard heuristic for cost-aware Bayesian optimization.
    cost_safe = np.maximum(cost, 1e-8)
    
    # For minimization, we want lower function values.
    # Thompson Sampling typically minimizes the sample.
    # To fit the "higher score = evaluate next" interface, we can negate the sample.
    # However, the cost division needs care.
    # If we want to minimize cost * value, or value / cost?
    # Usually, we want to maximize the "value per cost" or minimize "cost per value".
    # Let's stick to the convention of the previous program: score = -sample / cost.
    # This means we prefer low function values and low costs.
    
    scores = -samples / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
