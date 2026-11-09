"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and potentially cache expensive computations if needed
# though for 2000 points, direct computation is likely fast enough.
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s limit, keep 10s margin
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    # Ensure sigma is non-negative and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware Thompson Sampling
    # We want to minimize f(x) / cost(x) roughly, or rather select x that minimizes f(x) 
    # while being efficient with cost.
    # Standard Thompson Sampling: sample f(x) ~ N(mu, sigma^2), pick argmin.
    # Here we have a budget constraint implicit in the loop, but the acquisition 
    # function scores candidates.
    
    # The cost is 1 + mean(x). Since x is not passed, we cannot compute cost per candidate 
    # inside acquisition unless we infer it or the 'cost' vector provided is the cost 
    # of evaluating that specific candidate.
    # The prompt says: "mu, sigma and cost have shape (2000,)". 
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # The loop scores 2000 new uniform candidates with acquisition and evaluates the best one.
    # So `cost` vector likely contains the cost associated with each of the 2000 candidates.
    
    # Strategy: Sample a value for each candidate from its GP posterior.
    # To make it cost-aware, we can penalize the sampled value by the cost, 
    # or sample from a distribution that reflects cost-efficiency.
    # A common approach for cost-aware BO is to maximize E[I(x)] / cost(x).
    # For Thompson Sampling, we can sample f(x) and then score by -f_sampled / cost(x) 
    # or simply -f_sampled if we assume the loop handles cost by stopping.
    # However, the prompt says "Return 2000 finite scores (higher = evaluate next)".
    # If we just return -f_sampled, we ignore cost in the selection of the *next* point,
    # but the loop stops when total cost > 30.
    # To be "cost-aware" in the acquisition itself, we should prefer points that
    # offer good improvement relative to their cost.
    
    # Let's use a cost-aware sampling score:
    # Score = -f_sampled / cost
    # This approximates maximizing expected improvement per unit cost in a stochastic way.
    
    # Generate random samples from the GP posterior
    # mu and sigma are in objective units (minimization)
    samples = mu + sigma * np.random.randn(len(mu))
    
    # The cost vector is provided. We divide by cost to prioritize cheaper evaluations
    # that are expected to be good.
    # Note: If cost is very low, this amplifies the score. If cost is high, it dampens.
    # Since we minimize f, lower samples are better. We want higher scores for selection.
    # So we want -samples / cost to be high.
    # i.e., samples should be low (good) and cost should be low (cheap).
    
    # Safety check for cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -samples / cost_safe
    
    return scores
# EVOLVE-BLOCK-END
