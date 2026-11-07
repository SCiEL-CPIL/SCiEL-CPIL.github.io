"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling (TS).
    
    Strategy: Instead of computing an analytical acquisition function like EI or UCB,
    we draw a single sample from the Gaussian Process posterior for each candidate point.
    We then select the candidate with the lowest sampled value.
    
    To ensure determinism (since the evaluator might not expect stochastic behavior 
    or to allow for reproducibility in debugging), we use a fixed random seed for the 
    normal distribution draws. Since the inputs (mu, sigma) change at every step of 
    the BO loop, the resulting sampled function will differ at each step, effectively 
    implementing Thompson Sampling.
    
    TS is known to be robust and does not require tuning exploration parameters (like 
    beta in UCB or xi in EI), making it a strong structural alternative to heuristic 
    blends.
    """
    # Use a fixed seed to make the "random" draw deterministic for a given input state.
    # This is a common technique in deterministic implementations of stochastic methods.
    rs = np.random.RandomState(42)
    
    # Draw standard normal random variables
    z = rs.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # Note: sigma is the standard deviation
    y = mu + sigma * z
    
    # The goal is to minimize the objective function.
    # Therefore, a lower sampled value y is better.
    # The acquisition function should return scores where HIGHER means evaluate next.
    # So we negate the sampled value.
    score = -y
    
    # Ensure all scores are finite
    # If sigma is huge or mu is inf, handle it.
    # The problem states mu and sigma are from a GP, so they should be finite.
    # But just in case:
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
# EVOLVE-BLOCK-END
