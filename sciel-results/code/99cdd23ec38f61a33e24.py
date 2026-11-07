"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
        We return -sampled_values so that the minimum of the sampled function
        corresponds to the maximum score.
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to simple EI if we are running out of time, 
        # though this is unlikely for a single vectorized operation
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Thompson Sampling: Sample from the posterior predictive distribution
    # f(x) ~ N(mu, sigma^2)
    # We want to minimize f(x), so we maximize -f(x)
    # Generate a random sample for each candidate
    # np.random.normal is vectorized and fast
    
    # Use a fixed seed? No, the evaluator handles seeds by restarting the process or similar.
    # However, the task says "for each seed 0, 1 and 2 the evaluator starts...".
    # Usually, this implies the environment sets the random seed or the program should be stochastic.
    # If the program is deterministic, it will behave the same for all seeds unless the input changes.
    # The input (mu, sigma) changes based on the previous evaluations, which depend on the random initial points.
    # So the stochasticity of TS is crucial to explore different regions in different runs.
    
    # Sample from N(mu, sigma)
    # To ensure we don't have issues with 0 variance, we can add a tiny epsilon to sigma
    # but the problem states sigma is standard deviation.
    samples = np.random.normal(mu, sigma + 1e-9)
    
    # We want to pick the point with the lowest value.
    # The acquisition function returns scores where higher is better.
    # So we return -samples.
    return -samples
# EVOLVE-BLOCK-END
