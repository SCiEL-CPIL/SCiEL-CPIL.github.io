"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Draw a sample from the posterior predictive distribution for each candidate.
       Since the GP posterior is Gaussian, the sample is y = mu + sigma * z, where z ~ N(0, 1).
    2. To handle the cost, we normalize the sampled objective value by the cost.
       Since we are minimizing, we want low values. To convert to a "higher is better" score,
       we take the negative of the normalized value: score = -y / cost.
    3. This single sample captures both the mean (exploitation) and variance (exploration).
       High variance regions have a higher probability of yielding a very low (good) sample,
       thus promoting exploration.
    4. The cost normalization ensures that if two points have similar expected improvements,
       the cheaper one is preferred.
    
    This approach is stochastic, which helps in escaping local minima in multimodal functions
    like Rastrigin, unlike deterministic EI/UCB which can get stuck.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s kill

    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # f_best is a scalar, but Thompson Sampling doesn't strictly need it 
    # for the selection step itself, though it's good for sanity checks.
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Generate a random normal sample z
    # Using a fixed seed per call is not possible as we don't have state,
    # but we can use the global numpy random state which is re-seeded by the evaluator?
    # Actually, the evaluator likely re-seeds or uses a fresh process. 
    # Standard practice is to just use np.random.randn.
    z = np.random.randn(n)
    
    # Sample from the posterior predictive
    # y ~ N(mu, sigma^2)
    y_sample = mu + sigma * z
    
    # Cost-aware score: We want to minimize y_sample / cost.
    # So we return -y_sample / cost to maximize the score.
    score = -y_sample / cost
    
    # Check time
    if time.time() - start_time > deadline:
        # Fallback if somehow slow (shouldn't happen for vectorized ops)
        return np.zeros(n)
        
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

# EVOLVE-BLOCK-END
