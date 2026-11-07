import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

# We need a stable random seed for Thompson Sampling draws.
# Using a fixed seed ensures reproducibility within a single run,
# but different runs might benefit from different seeds.
# However, since the function is called repeatedly, we should use a
# counter or a persistent random state to get different samples each time.
# But the task says "Return 2000 finite scores". It doesn't say they must be deterministic.
# Using a global RNG is fine.

_rng = np.random.default_rng(42)

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Thompson Sampling.
    
    We draw a single sample from the GP posterior: y = mu - sigma * z,
    where z is drawn from a standard normal distribution.
    We then score candidates by -y / cost (minimize y, so maximize -y/cost).
    This treats the objective as a stochastic reward-to-go ratio.
    
    To adapt the exploration level, we can scale the sigma term.
    However, standard Thompson Sampling already balances exploration/exploitation
    based on the uncertainty (sigma). Early on, sigma is large, so the sample
    varies a lot (exploration). Later, sigma is small near known points,
    so the sample is close to mu (exploitation).
    
    We divide by cost to account for the cost-awareness.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to pure exploitation if running out of time
        scores = -np.asarray(mu, dtype=np.float64) / np.maximum(np.asarray(cost, dtype=np.float64), 1e-8)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # Draw a single sample from the posterior
    # z ~ N(0, 1)
    z = _rng.standard_normal(mu.shape)
    
    # y = mu - sigma * z  (minimization problem, so lower is better)
    # This sample represents a possible value of the function at each point.
    y = mu - sigma * z
    
    # Score: we want to minimize y, so we maximize -y.
    # Cost-awareness: divide by cost.
    # score = -y / cost
    scores = -y / cost

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
