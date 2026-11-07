import numpy as np
import time
from scipy.stats import norm

# Safety margin: The evaluator kills at 330s.
# We set a deadline at 325s to allow for cleanup/return.
_DEADLINE = time.time() + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviation predictions.
    f_best: float, the best observed value so far.
    cost: (2000,) array of costs for each candidate.
    
    Returns:
    scores: (2000,) array of finite scores. Higher is better.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros_like(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Ensure inputs are finite and valid
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # 1. Calculate Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    
    # Standard EI formula
    EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)

    # 2. Structural Change: Hard Cost Gate
    # Instead of smooth cost-weighting, use a hard gate based on cost efficiency.
    # Calculate the average cost of the top 10% EI candidates.
    # Candidates with cost below this threshold get full EI score.
    # Others get heavily damped.
    
    # Find the threshold: 90th percentile of costs among top EI candidates
    # Or simpler: use the mean cost of all candidates as a baseline,
    # but weight by how "promising" they are.
    
    # Let's use a dynamic threshold based on the distribution of costs.
    # If a candidate's cost is less than the median cost, it's "cheap".
    # If it's expensive, dampen its score.
    
    # But the inspiration says: "keep full EI only when a candidate's predicted cost 
    # is below the running average cost-per-unit-improvement, heavily damping the rest"
    
    # Cost-per-unit-improvement is roughly cost / EI. But EI varies.
    # Let's define a "value" metric: EI / cost.
    # We want to favor high EI/cost.
    
    # Structural change: Use a sigmoid gate on (cost - threshold)
    # Where threshold is the median cost of the top 20% EI candidates.
    
    # Get top 20% EI candidates
    n_top = max(1, int(0.2 * len(EI)))
    top_ei_indices = np.argsort(EI)[-n_top:]
    top_costs = cost[top_ei_indices]
    
    # Threshold: median cost of top EI candidates
    threshold = np.median(top_costs)
    
    # Gate: sigmoid function that is 1 for cost < threshold, 0 for cost >> threshold
    # Steepness controls how hard the gate is
    steepness = 5.0
    gate = 1.0 / (1.0 + np.exp(steepness * (cost - threshold) / (threshold + 1e-6)))
    
    # Apply gate to EI
    gated_ei = EI * gate
    
    # Add a small exploration term for high sigma, low cost areas
    # UCB-like term with cost damping
    beta = 1.5
    cost_damping = np.power(cost / (np.mean(cost) + 1e-6), 0.5)
    ucb_term = beta * sigma / cost_damping
    
    # Combine: Primary signal is gated EI, secondary is UCB
    # Weighted sum
    w_primary = 1.0
    w_secondary = 0.5
    
    scores = w_primary * gated_ei + w_secondary * ucb_term
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small floor to avoid ties or zero scores for valid candidates
    scores = np.maximum(scores, 1e-10)
    
    return scores
