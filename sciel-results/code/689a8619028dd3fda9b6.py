# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere function.
    Uses Probability of Improvement (PI) which is smoother and less prone
    to over-exploitation than EI. For 1D sphere with known global minimum at 0,
    PI provides a good balance: it favors points likely to beat f_best but
    with less aggressive exploitation than EI.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is positive for stable computation
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Probability of Improvement (PI)
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (fb - mu) / sigma_safe
    pi_score = norm.cdf(z)

    # For sphere in [-5,5], the global minimum is at x=0 with value 0.
    # We can use this knowledge: if f_best is already very small,
    # we should focus on exploitation near the current best.
    # If f_best is large, we need more exploration.
    
    # Add a small penalty for high uncertainty to avoid exploring
    # regions with very high sigma (unless f_best is still large)
    # This is a soft constraint: we don't want to completely avoid
    # exploration, but we want to be cautious about very uncertain regions.
    
    # Adaptive exploration: when f_best is large, allow more exploration
    # by reducing the penalty on high sigma regions.
    # When f_best is small, increase the penalty to focus on exploitation.
    
    # Scale: f_best ranges from 0 to 25 (sphere in [-5,5])
    # Use a smooth transition
    fb_norm = np.clip(fb / 25.0, 0, 1)
    
    # Exploration weight: higher when f_best is large
    # At fb=25 (initial), we want more exploration
    # At fb=0 (optimal), we want pure exploitation
    exploration_weight = 0.1 * (1.0 - fb_norm)
    
    # Uncertainty penalty: subtract a fraction of sigma
    # This discourages very uncertain predictions
    # But the weight decreases as we get closer to the optimum
    uncertainty_penalty = exploration_weight * sigma_safe
    
    # Final score: PI minus uncertainty penalty
    # This gives a smooth acquisition function that balances
    # probability of improvement with uncertainty
    score = pi_score - uncertainty_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If time is running out, return immediately
    if time.time() > _DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
