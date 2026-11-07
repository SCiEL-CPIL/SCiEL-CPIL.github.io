"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Strategy:
    - Sample from the posterior predictive distribution at each candidate point.
    - For minimization, we want low values, so we sample y = mu + sigma * z
      and then maximize -y (i.e., minimize y).
    - This provides stochastic exploration that adapts to the remaining uncertainty.
    - To make the sampling deterministic per call (since we need to return a single
      set of scores for 2000 candidates), we use a fixed random seed based on
      a global counter or hash of the input to ensure reproducibility within a step.
    - Actually, for Thompson Sampling in a batch setting where we pick the best
      candidate from 2000, we can simply sample one value per candidate and
      select the one with the lowest sampled value. But here we return scores
      for all 2000 candidates. The caller picks the max score.
    - So, score = -sampled_value.
    - To add exploration, we sample z ~ N(0, 1) for each candidate.
    - However, pure Thompson sampling can be too greedy or too random.
    - A common trick is to use a deterministic proxy for the sample, e.g.,
      sampling from a distribution that balances exploitation and exploration.
    - Let's stick to standard Thompson Sampling: sample y_i = mu_i + sigma_i * z_i
      where z_i ~ N(0,1). Then score_i = -y_i.
    - To avoid the randomness causing inconsistency between calls if the GP
      changes, we can use a fixed seed for the random number generator.
      But the GP parameters (mu, sigma) change each step.
      Using a fixed seed for the z-samples ensures that the relative ordering
      of candidates is driven by the mu and sigma changes, not by random noise
      that changes every step. This makes the exploration more stable.
      
    - Let's use a global counter to seed the RNG, or just use a fixed seed.
      If we use a fixed seed, the z-samples are the same every time.
      Then score_i = -(mu_i + sigma_i * z_i).
      This is equivalent to adding a constant vector z to the mu/sigma combination.
      This might not be ideal.
      
    - Alternative: Use a deterministic approximation of Thompson Sampling.
      One way is to use the mean of the posterior predictive distribution
      plus a term that encourages exploration. But that's just UCB.
      
    - Let's try a different angle: The directive suggests "approximating the posterior
      predictive variance at each candidate and adding a small amount of noise to the mean".
      This sounds like a randomized scoring function.
      
    - Let's implement a simple randomized score:
      score = -mu + sigma * z
      where z is a fixed vector of random normal numbers.
      By using a fixed z, we create a deterministic mapping from (mu, sigma) to scores,
      which is effectively a randomized linear combination of exploitation and exploration.
      This is a form of Thompson Sampling if we interpret the fixed z as the realized
      sample from the posterior over the function values.
      
    - To make it more adaptive, we can scale the noise based on the global uncertainty.
      But let's start with the simple version.
      
    - We need to ensure the scores are finite.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    n = len(mu)
    
    # Use a fixed seed for the random samples to ensure consistency
    # across calls with the same mu/sigma structure, but different values.
    # This makes the exploration deterministic relative to the GP state.
    rng = np.random.RandomState(42)
    z = rng.normal(0, 1, n)
    
    # Thompson Sampling score for minimization:
    # We want to minimize y, so we maximize -y.
    # y = mu + sigma * z
    # score = -y = -mu - sigma * z
    
    # To balance exploration, we can scale sigma.
    # A common choice is to use the raw sigma, or a scaled version.
    # Let's try raw sigma first.
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    scores = -mu - sigma_safe * z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
