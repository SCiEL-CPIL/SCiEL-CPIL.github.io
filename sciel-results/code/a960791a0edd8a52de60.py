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
    
    This function computes a stochastic acquisition score by drawing a sample
    from the GP posterior for each candidate and dividing by the evaluation cost.
    This encourages exploration by allowing candidates with high uncertainty to
    occasionally be selected, while the cost division ensures efficiency.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        return np.random.uniform(size=mu.shape)

    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a single sample from the GP posterior for each candidate
    # f_sample ~ N(mu, sigma^2)
    # To make the score "higher is better" for minimization, we use -f_sample
    # But we also need to account for cost.
    # The task says "higher = evaluate next".
    # If we minimize f(x) subject to cost, we want to maximize (improvement / cost) or minimize (f_sample / cost).
    # Since the return value must be "higher is better", we can return - (f_sample / cost).
    # However, f_sample can be large positive.
    # Let's stick to the standard Thompson Sampling for minimization:
    # We want to select the candidate that minimizes f_sample / cost.
    # So the acquisition score should be proportional to - (f_sample / cost).
    
    # To ensure numerical stability and avoid extreme outliers dominating,
    # we can clip the sample or just rely on the fact that the evaluator picks the max.
    # The evaluator picks the candidate with the MAXIMUM score.
    # So we want the candidate with the MINIMUM (f_sample / cost) to have the MAXIMUM score.
    # Therefore: score = - (f_sample / cost)
    
    # Generate random noise
    noise = np.random.standard_normal(mu.shape)
    
    # Compute sampled objective values
    f_sample = mu + sigma_safe * noise
    
    # Compute cost-aware score
    cost_safe = np.maximum(cost, 1e-8)
    scores = - (f_sample / cost_safe)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
