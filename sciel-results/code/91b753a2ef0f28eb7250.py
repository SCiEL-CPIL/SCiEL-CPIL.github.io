"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 325.0  # 5 seconds safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Compute Thompson Sampling acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Check deadline
    if time.time() > DEADLINE:
        # Return a safe fallback: random or uniform scores
        return np.ones_like(mu, dtype=np.float64) * 1e-10
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Thompson Sampling:
    # Sample a single value from the predictive distribution for each candidate.
    # Since we need to return scores for the loop to select the "best" one,
    # and the loop evaluates the argmax of the returned scores,
    # we can simulate the Thompson Sampling decision by generating a random draw
    # and returning the cost-adjusted sampled value.
    
    # However, standard Thompson Sampling usually involves drawing a function
    # and evaluating it. Here, the interface returns scores for 2000 points.
    # The caller will pick argmax(scores).
    # To emulate TS, we sample y_i ~ N(mu_i, sigma_i^2) for each point i.
    # We want to maximize -y_i / cost_i (since lower y is better, and we want to minimize cost).
    # Or maximize (f_best - y_i) / cost_i?
    # Actually, TS for minimization: we want to find the point with the lowest y.
    # So we sample y_i. The "score" for selection should be high for points that
    # are likely to have low y and low cost.
    # A common cost-aware TS score is -y_i / cost_i.
    # Let's use a fixed seed for the random numbers to ensure determinism?
    # The task says "seeded instances", but the acquisition function itself
    # might be called multiple times. If we use a fixed seed inside,
    # the noise will be the same every time, which might be bad if the GP changes.
    # However, the GP parameters (mu, sigma) change each iteration.
    # If we use a fixed seed, the noise vector is constant.
    # Let's use a seed based on the current state to vary it slightly?
    # Or just use a global counter?
    # Given the constraint "deterministic" is not explicitly required for the *internal* randomness
    # of the acquisition, but reproducibility of the *result* might be preferred.
    # However, standard BO loops are deterministic if the acquisition is deterministic.
    # TS is inherently stochastic.
    # Let's use a fixed seed for the noise to make the function deterministic given the inputs.
    # This means the "random" draw is actually a fixed vector of standard normal variates.
    
    # Generate fixed standard normal noise for 2000 points
    # Using a fixed seed ensures that if mu/sigma/cost are the same, the output is the same.
    # This is important for reproducibility in the evaluation harness if it re-runs.
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(2000)
    
    # Sampled values
    sampled_values = mu + sigma * noise
    
    # We want to minimize the function value.
    # Cost-aware: we prefer points with low sampled value and low cost.
    # Score = -sampled_value / cost
    # Higher score is better.
    # If sampled_value is very negative (good), score is large positive.
    # If cost is high, score is smaller.
    
    scores = -sampled_values / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip extremely large values to avoid numerical issues in argmax or downstream
    # though argmax is robust, let's be safe.
    # The magnitude of scores depends on mu/sigma scale.
    # Rosenbrock values can be large.
    
    return scores
# EVOLVE-BLOCK-END
