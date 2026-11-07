"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Aware Thompson Sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute constants
_EPS = 1e-9
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample a posterior predictive value for each candidate point:
       f_sample(x) = mu(x) + sigma(x) * z(x), where z(x) ~ N(0, 1)
       
    2. Compute the improvement over the current best for this sample:
       I_sample(x) = max(0, f_best - f_sample(x))
       
    3. Normalize by the evaluation cost to prioritize cheap improvements.
       Score(x) = I_sample(x) / cost(x)
       
    This approach (Thompson Sampling) is structurally different from UCB/EI
    as it uses a stochastic sample from the posterior rather than a deterministic
    bound or expectation. This often leads to better exploration in early stages
    and more precise exploitation later, adapting the trade-off naturally.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Deadline guard
    t0 = time.time()
    
    # 1. Sample from the posterior predictive distribution
    # Use a fixed seed relative to the call count? No, we don't have state.
    # Standard Thompson Sampling uses random sampling each time.
    # However, for reproducibility in some contexts, one might worry, but the task
    # implies a loop. Randomness is fine and expected for TS.
    # We generate standard normal random numbers.
    z = np.random.standard_normal(n)
    
    # f_sample = mu + sigma * z
    # Ensure sigma is non-negative to avoid sign flips if numerical errors occur
    sigma_safe = np.maximum(sigma, 0.0)
    f_sample = mu + sigma_safe * z
    
    # 2. Compute Potential Improvement for this sample
    # Improvement is positive if the sampled value is better than f_best
    improvement = f_best - f_sample
    improvement = np.maximum(improvement, 0.0)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_MARGIN:
        cost_safe = np.maximum(cost, _EPS)
        return improvement / cost_safe

    # 3. Cost-awareness
    # Divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, _EPS)
    
    final_score = improvement / cost_safe
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Final deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return final_score
        
    return final_score

# EVOLVE-BLOCK-END
