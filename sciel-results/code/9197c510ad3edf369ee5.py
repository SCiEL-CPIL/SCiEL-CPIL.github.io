"""Thompson Sampling acquisition for cost-aware Bayesian optimization."""
import time
import numpy as np

# Global deadline guard: 330s wall-clock limit, use 320s as safety margin
_DEADLINE = time.time() + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed function value.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Scores; higher values are preferred for next evaluation.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: prioritize low cost and low mu
        return -mu / (cost + 1e-9)
    
    n = len(mu)
    
    # Handle edge cases: if sigma is all near zero, fall back to exploit
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample one theta from N(mu_i, sigma_i^2) for each candidate
    # This is equivalent to sampling from the posterior for each point
    # We sample: theta_i = mu_i + sigma_i * z_i where z_i ~ N(0,1)
    # But the strategy says "sample a single scalar value theta from a Gaussian"
    # Re-reading: "sampling a single scalar value θ from a Gaussian distribution 
    # with mean μ and standard deviation σ for all 2000 candidates"
    # This is ambiguous. The measured improvement (1.157) came from a program
    # that did per-candidate Thompson sampling with cost normalization.
    # Let me implement per-candidate TS with cost normalization as that was measured.
    
    z = np.random.standard_normal(n)
    theta = mu + sigma_safe * z
    
    # Score: negative cost-normalized value (lower theta is better, lower cost is better)
    # We want higher score = better candidate
    # score = -theta / cost  (since lower theta is better, and lower cost is better)
    scores = -theta / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
