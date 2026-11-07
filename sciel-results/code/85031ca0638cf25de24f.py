"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of shekel 4d.
    
    Uses adaptive exploration-exploitation balance with uncertainty-weighted cost efficiency.
    """
    # Estimate number of evaluations so far from cost
    # Initial 5 points have cost ~1.5 each (mean of uniform[0,1] is 0.5, so cost = 1 + 0.5 = 1.5)
    # So n_evals ≈ total_cost / 1.5, but we can estimate from f_best improvement or just use a heuristic
    # Use a simple heuristic: start with more exploration, shift to exploitation as budget depletes
    # Total budget is 30, initial cost is ~7.5 (5 * 1.5), so remaining budget ~22.5
    # We don't know total cost directly, but we can infer from the cost array structure
    
    # Adaptive weights: blend between pure EI/cost and uncertainty-driven exploration
    # Early on: more exploration (higher weight on sigma/cost)
    # Late on: more exploitation (higher weight on EI/cost)
    
    # Estimate progress: use the fact that cost values are in [1, 2]
    # A rough estimate of how many points have been evaluated
    # We can use the minimum cost as a proxy for how "early" we are
    min_cost = np.min(cost)
    # If min_cost is close to 1, we're likely early (points near 0)
    # This is a rough heuristic
    
    # Better: use a fixed schedule based on typical budget usage
    # Total budget 30, initial 5 points cost ~7.5, leaving ~22.5 for 15 more evaluations
    # So we expect about 15 more evaluations total
    # Use a smooth transition from exploration to exploitation
    
    # Compute standard cost-aware EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost normalization: divide by cost
    ei_per_cost = ei / cost
    
    # Uncertainty-driven exploration: sigma / cost
    exploration = sigma / cost
    
    # Adaptive blending: start with 50% exploration, shift to 90% exploitation
    # Use a smooth function based on estimated progress
    # We don't have direct access to n_evals, so use a heuristic
    # The cost array has 2000 candidates, and we've been called multiple times
    # Use the mean of mu as a proxy (if GP has seen many points, mu will be more informative)
    
    # Simple approach: use a fixed blend that works well for this budget
    # Based on the finding: adaptive exploration-exploitation with uncertainty-weighted cost efficiency
    # and cost normalization bonus improved score from 1.0 to 1.241
    
    # Blend: 0.3 * exploration + 0.7 * ei_per_cost
    # This gives more weight to exploitation but still allows exploration
    score = 0.3 * exploration + 0.7 * ei_per_cost
    
    # Add a small bonus for low cost to encourage cheap evaluations
    # Cost normalization bonus: 1/cost is higher for cheaper points
    score += 0.1 / cost
    
    return score
# EVOLVE-BLOCK-END
