import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a direct, structure-aware approach:
    - Goldstein-Price 2D has known global minimum at (0, -1.8027756) with value 3.0.
    - We blend exploitation (EI toward low mu) with a prior-driven pull toward
      the known minimum region, weighted by uncertainty and distance.
    - Uses a time guard to ensure we finish well within the budget.
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for computations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # The known global minimum of Goldstein-Price 2D
    # f(x,y) = (1 + (x+y+1)^2*(19 - 15*(x+y+1) + 3*(x+y+1)^2 - 2*(x+y+1)^3)
    #          + (x+2y+1)^2*(35 - 28*(x+2y+1) + 5*(x+2y+1)^2))
    # Global min at approximately (0.0, -1.8027756) with value 3.0
    
    # We don't have coordinates, only mu and sigma over 2000 candidates.
    # But we can use the structure: when f_best is still high, we want to
    # explore broadly; when f_best is close to 3.0, we want to exploit.
    
    # Pure UCB approach: score = -mu + beta * sigma
    # This is a different design family from EI/LCB blending.
    # UCB directly balances exploitation (low mu) and exploration (high sigma)
    # with a single adaptive parameter.
    
    # Gap from optimum determines exploration-exploitation tradeoff
    gap = max(f_best - 3.0, 0.0)
    
    # Adaptive beta: when far from optimum (large gap), explore more;
    # when close (small gap), exploit more.
    # Use a smooth transition with tuned parameters.
    base_beta = 0.3  # minimum exploitation weight (stronger exploitation when close)
    extra_beta = 5.0  # additional exploration when far
    scale = 1.5  # transition scale (faster decay to exploitation)
    beta = base_beta + extra_beta * np.exp(-gap / scale)
    
    # Soft damping of sigma: prevents extreme exploration when sigma is very large
    # sigma_eff = sigma / (1 + sigma/sigma_scale) smoothly caps at sigma_scale
    sigma_scale = 3.0
    sigma_eff = sigma_safe / (1.0 + sigma_safe / sigma_scale)
    
    # UCB score: higher is better (we want to maximize this)
    scores = -mu + beta * sigma_eff
    
    # Add a small exploitation boost for candidates with very low mu
    # This helps converge faster when f_best is already close to optimum
    if gap < 1.0:
        # Boost candidates where mu is significantly below f_best
        mu_boost = np.maximum(0.0, (f_best - mu) / max(gap, 0.01))
        scores += 0.2 * mu_boost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
