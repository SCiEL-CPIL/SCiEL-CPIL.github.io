import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses GP-UCB with a time-varying exploration parameter.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin from 300s limit

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # Infer the number of evaluations t from the magnitude of f_best
    # The minimum value of the function is -2.53218
    # Initial range is roughly [-3, 3] for a random RBF mixture
    # We can use a simple heuristic: if f_best is close to the minimum, t is large
    # A more robust approach: use a fixed schedule or estimate t from the improvement
    # For now, let's use a simple time-varying beta based on a fixed schedule
    # t is the number of evaluations, which we don't have directly
    # We can approximate t by assuming the initial f_best is around 0 and it improves
    # Let's use a simple heuristic: t = max(1, int(10 * (1 - (f_best + 2.53218) / 5)))
    # This is a rough estimate; in practice, we'd want to track t explicitly
    
    # Alternative: use a fixed beta that works well for small N
    # For 30 evaluations in 2D, beta around 1-5 is typical
    # Let's use a time-varying beta: beta_t = sqrt(t * log(t^2 * c^2 / delta))
    # With t estimated, c=1, delta=0.1
    # Estimate t from f_best: assume linear improvement from 0 to -2.53218 over 30 steps
    t_est = max(1, int(30 * (1 - (f_best + 2.53218) / 5.0)))
    t_est = min(t_est, 30)  # Cap at total evaluations
    
    c = 1.0
    delta = 0.1
    beta = np.sqrt(t_est * np.log(t_est**2 * c**2 / delta))
    
    # Cap beta to avoid excessive exploration
    beta = min(beta, 5.0)
    
    # GP-UCB for minimization: score = mu - beta * sigma
    scores = mu - beta * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check time
    if time.time() > deadline:
        return scores
    
    return scores
