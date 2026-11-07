"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5s margin before 330s kill


def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just return mu-based scores (lower mu = better)
        return -mu
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive threshold for Expected Improvement
    # Scale the improvement target by the current best value to balance
    # exploration (when f_best is large) and exploitation (when f_best is small)
    # We want to find points where the expected improvement over f_best is high.
    # Since we minimize, improvement = f_best - f(x).
    # E[improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    # To dynamically balance, we can scale the "improvement" we seek.
    # A common variant is to use a threshold xi > 0, seeking improvement over f_best - xi.
    # Here we let xi be proportional to |f_best| to adapt to the scale of the function.
    # For Weierstrass, f_best starts around 1-2 and goes to 0.
    # Let's use xi = 0.1 * abs(f_best) + 1e-6 to ensure some exploration.
    xi = 0.1 * abs(f_best) + 1e-6
    
    # Target improvement over (f_best - xi)
    # Let mu_target = mu, sigma_target = sigma
    # We want E[max(0, (f_best - xi) - mu)]
    
    z = (f_best - xi - mu) / sigma
    
    # Avoid numerical issues
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - xi - mu) * Phi_z + sigma * phi_z
    
    # Higher EI is better
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
