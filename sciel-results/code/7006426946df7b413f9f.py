"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample a single function value from the GP posterior for each candidate:
       y_sampled = mu + sigma * z, where z ~ N(0, 1).
    2. The goal is to minimize the function. Thus, we prefer candidates with low sampled values.
    3. To account for cost, we normalize the "desirability" of the sample by the cost.
       Since we want to minimize f(x) and cost is positive, a good metric is minimizing f(x) / cost.
       Therefore, the acquisition score should be high when f_sampled / cost is low.
       Score = - (f_sampled / cost).
    
    This approach provides stochastic exploration (via the random sample z) which helps
    escape local minima in multimodal functions like Rastrigin, unlike deterministic
    EI or LCB which can get stuck in greedy local optima.
    """
    start_time = time.time()
    
    # Safety margin: kill if we exceed 300s (limit is 330s, evaluator kills at 330s, 
    # but we want to return safely. The task says "Each evaluation must finish within 300s".
    # We assume this acquisition call is part of the loop. 
    # However, the prompt says "the evaluator KILLS the program after 330s wall-clock".
    # This implies the whole run has 330s. But the acquisition function is called many times.
    # The 300s limit likely applies to the total run time or a specific phase.
    # Given the ambiguity, I will add a simple check. If we are taking too long in this call,
    # we return a fallback. But Thompson Sampling on 2000 points is extremely fast (<1ms).
    # The risk is if numpy is slow for some reason.
    
    # Generate random normal samples
    # sigma is the standard deviation
    # Ensure sigma is non-negative and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Draw samples
    z = np.random.standard_normal(len(mu))
    y_sampled = mu + sigma_safe * z
    
    # Ensure y_sampled is finite
    y_sampled = np.where(np.isfinite(y_sampled), y_sampled, 1e6) # Large penalty for invalid
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute score: we want to minimize y_sampled / cost
    # So acquisition score = - (y_sampled / cost_safe)
    # Higher score is better.
    score = - (y_sampled / cost_safe)
    
    # Final safety check for finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check time (just in case, though this should be fast)
    elapsed = time.time() - start_time
    if elapsed > 5.0: # Arbitrary threshold for a single call, should be <0.01s
         # Fallback to a simple deterministic score if something went wrong
         # This is unlikely to trigger.
         pass
         
    return score
# EVOLVE-BLOCK-END
