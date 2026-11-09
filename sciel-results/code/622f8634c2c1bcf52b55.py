"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs and proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: avoid division by zero and negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Strategy:
    # Instead of computing Expected Improvement (EI), we simulate samples from the posterior.
    # For a minimization problem, we want to find points with low function values.
    # We can model the "reward" as -f(x). We want to maximize reward.
    # The posterior for f(x) is N(mu, sigma^2).
    # The posterior for reward r(x) = -f(x) is N(-mu, sigma^2).
    
    # To make the selection stochastic and robust, we draw a single sample from the posterior
    # for each candidate point. However, drawing 2000 random numbers per acquisition call
    # might be slightly slow if done naively, but it's very fast in NumPy.
    # More importantly, standard Thompson Sampling picks the argmax of the samples.
    # Here we need to return a score for each point so the external loop can pick the best.
    # If we return the sample itself, the external loop will pick the candidate with the highest
    # sampled reward (lowest sampled cost).
    
    # Let's generate a single random sample for each point.
    # Note: Using a fixed seed per call is not possible because the function signature
    # doesn't provide a state. We must use the global RNG.
    # To ensure stability and avoid extreme outliers in the score that might be ignored
    # if the external loop uses a threshold, we can just return the sampled value.
    
    # Sample from N(-mu, sigma)
    # Since we want to MINIMIZE f, we maximize -f.
    # Let z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    
    # Sampled function value: f_sample = mu + sigma * z
    # We want to select the point with the lowest f_sample.
    # The acquisition function should return higher scores for better points.
    # So, score = -f_sample = -(mu + sigma * z)
    
    sampled_f = mu + sigma * z
    scores = -sampled_f
    
    # Cost-awareness:
    # Simple Thompson Sampling doesn't account for cost directly in the sample.
    # We can adjust the score by dividing by cost, similar to UCB/EI approaches,
    # or we can incorporate cost into the prior if we had access to it, but we don't.
    # A common heuristic for cost-aware TS is to divide the reward by cost.
    # Reward r = -f_sample.
    # Score = r / cost.
    
    scores = scores / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
