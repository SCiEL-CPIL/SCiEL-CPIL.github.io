"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with UCB fallback.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher means more desirable to evaluate
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from posterior of improvement
    # For minimization, we want to find points where f(x) < f_best
    # The posterior for f(x) is N(mu, sigma^2)
    # We sample a single realization from the posterior
    rng = np.random.default_rng(seed=42)
    
    # Sample from the GP posterior
    # Add a small exploration bonus to encourage exploration
    exploration_bonus = 0.1  # Scale relative to typical EI magnitudes
    
    # Thompson Sampling score: sample from posterior, lower is better
    # But we want higher score = more desirable, so negate
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # The "improvement" is f_best - f(x), so we want to maximize this
    # In Thompson sampling, we sample f(x) and then the improvement is f_best - sample
    # Higher improvement = more desirable
    improvement_samples = f_best - samples
    
    # Add exploration bonus weighted by uncertainty
    # This helps explore regions with high uncertainty
    uncertainty_weight = 0.5
    exploration_term = uncertainty_weight * sigma_safe
    
    # Combine: primary signal is the sampled improvement, secondary is uncertainty
    ts_scores = improvement_samples + exploration_term
    
    # Cost-aware gating: penalize high-cost points
    # Cost is in [1, 2], so use exponential decay
    # k=0.5 gives exp(-0.5*1)=0.61, exp(-0.5*2)=0.37
    k = 0.5
    cost_gate = np.exp(-k * cost)
    
    # Apply cost gate multiplicatively
    scores = ts_scores * cost_gate
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
