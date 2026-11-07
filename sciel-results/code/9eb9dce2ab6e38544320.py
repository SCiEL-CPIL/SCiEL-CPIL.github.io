"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Structural Change: Replaces deterministic Expected Improvement (EI) with 
    stochastic Thompson Sampling. 
    - Samples a function f_s ~ N(mu, sigma^2)
    - Minimizes f_s * cost_penalty
    - This implicitly trades off exploration (high sigma -> high variance in f_s) 
      and exploitation (low mu) while penalizing high-cost evaluations.
    """
    start_time = time.time()
    
    # Ensure inputs are 1D arrays of same length
    n = len(mu)
    
    # Numerical stability for sigma
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Cost normalization: 
    # We want to penalize high cost. 
    # A multiplicative penalty on the objective is a standard way to incorporate cost 
    # into Thompson Sampling (minimizing cost-weighted regret).
    # Alternatively, we can view this as minimizing expected cost-weighted value.
    # Let's use a penalty factor that is proportional to cost.
    # To avoid extreme scaling, we normalize cost relative to the median or max.
    # Using max cost for normalization keeps the scale consistent.
    max_cost = np.max(cost)
    if max_cost <= 0:
        max_cost = 1.0
    
    # Normalize cost to [0, 1] range roughly, or just use it as a weight.
    # If we minimize f_s * cost, a point with f_s=10, cost=2 is worse than f_s=10, cost=1.
    # But we also want to explore. 
    # Let's try: score = - (sampled_value * cost_weight)
    # Where cost_weight scales the importance of cost.
    
    # A more robust approach for "cost-aware" in TS:
    # Minimize: sampled_value * (1 + alpha * normalized_cost)
    # Or simply: minimize sampled_value / cost? No, that rewards high cost if value is negative.
    # We want to MINIMIZE function value.
    # If cost is high, we should be less willing to accept a marginal improvement.
    # EI/cost approximates improvement per unit cost.
    # In TS, we don't have "improvement" directly, we have "value".
    # However, minimizing `value * cost` is a valid heuristic for cost-aware TS 
    # if we assume the baseline is 0. Since f_best is the reference, 
    # we can consider the "regret" of a sample: max(0, sampled_value - f_best).
    # Let's minimize: cost * max(0, sampled_value - f_best) + small * sampled_value
    # Actually, simpler: minimize sampled_value * cost_factor.
    # Let's use the same structural logic as the successful EI/cost:
    # The successful program used EI/cost. 
    # TS equivalent: Sample f_s. Score = -f_s / cost? 
    # If f_s is low (good), -f_s is high (good score).
    # If cost is high, -f_s/cost is smaller (bad score).
    # This aligns with "improvement per unit cost".
    
    # Let's sample from the posterior
    # z ~ N(0, 1)
    z = np.random.standard_normal(n)
    sampled_values = mu + sigma_safe * z
    
    # Cost penalty
    # We want higher scores for "better" candidates.
    # "Better" means: Low sampled value, Low cost.
    # Score = -sampled_values / cost_safe
    # Let's refine: 
    # If sampled_values < f_best, it's an improvement.
    # The magnitude of improvement is (f_best - sampled_values).
    # Cost-adjusted improvement: (f_best - sampled_values) / cost
    # This is exactly the TS version of the EI/cost heuristic!
    # EI is E[max(0, f_best - f)]. 
    # TS samples f. The "instantaneous improvement" is max(0, f_best - f).
    # So score = max(0, f_best - sampled_values) / cost
    
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate cost-aware improvement for this sample
    improvements = np.maximum(f_best - sampled_values, 0.0)
    scores = improvements / cost_safe
    
    # Add a small exploration bonus to ensure we don't get stuck if all improvements are 0
    # This helps in early stages or when the model is confident but wrong.
    # The successful program had a sigma/cost bonus. 
    # In TS, the variance is already sampled, but if the sample lands on a local max of the 
    # improvement function (i.e., f_best - sampled is small), we might stall.
    # Adding a small term proportional to sigma/cost encourages evaluating uncertain areas.
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Fallback to a simpler, faster heuristic if we are running out of time
        # Use a deterministic surrogate: -mu / cost
        simple_scores = -mu / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
        
    return scores
# EVOLVE-BLOCK-END
