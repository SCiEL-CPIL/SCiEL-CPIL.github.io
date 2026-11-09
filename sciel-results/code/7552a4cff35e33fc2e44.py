import numpy as np
from scipy.stats import norm
import time

# Deadline guard constants
_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Blended acquisition: Cost-normalized EI + Adaptive UCB exploration term.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
       EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
    2. Compute a UCB-like term for minimization: -mu + kappa * sigma.
       This encourages exploration in high-uncertainty regions.
    3. Blend the two: score = (EI + w_ucb * gamma * UCB) / cost.
       - EI provides principled probabilistic improvement (exploitation).
       - UCB ensures exploration, especially early on when f_best is high.
       - w_ucb is an adaptive weight that decays as f_best improves.
       - gamma scales the UCB contribution to be comparable to EI.
    4. Cost normalization accounts for the cost-aware setting.
    
    This combines the structural idea of the current program's focus on cost-awareness
    and exploration (via Thompson/UCB) with the refinement machinery of the top
    inspiration (blended EI + UCB with adaptive weights).
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback to a simple heuristic if we are running out of time
        # Return inverse cost (prefer cheap points)
        cost_arr = np.asarray(cost, dtype=np.float64)
        safe_cost = np.where(cost_arr < 1e-9, 1e-9, cost_arr)
        return 1.0 / safe_cost
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-9
    
    # Ensure sigma is positive and finite
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    
    # Ensure cost is positive and finite
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # --- Compute Expected Improvement (EI) ---
    z = (f_best - mu_safe) / safe_sigma
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z_clipped)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    ei = (f_best - mu_safe) * cdf_z + safe_sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # --- Compute UCB term ---
    # UCB = -mu + kappa * sigma
    # kappa should be adaptive: higher when f_best is high (early), lower when f_best is low (late)
    
    # Adaptive kappa: start around 1.5, decay to ~0.3
    # Use a smooth decay based on f_best
    # f_best for Powell 4D starts ~10-20, ends near 0
    try:
        kappa = 0.3 + 1.2 / (1.0 + np.exp((f_best - 3.0) * 0.8))
    except OverflowError:
        kappa = 0.3 if f_best < 3.0 else 1.5
    
    kappa = np.clip(kappa, 0.1, 2.0)
    
    ucb = -mu_safe + kappa * safe_sigma
    
    # --- Adaptive blend weight ---
    # w_ucb controls how much UCB contributes relative to EI
    # We want UCB weight HIGH when f_best is HIGH (early exploration).
    # w_ucb = 1 / (1 + exp((5 - f_best) * 0.5))
    # If f_best = 10: exp(-2.5) ~ 0.08, w ~ 0.92 (High)
    # If f_best = 0: exp(2.5) ~ 12, w ~ 0.08 (Low)
    
    try:
        w_ucb = 1.0 / (1.0 + np.exp((5.0 - f_best) * 0.5))
    except OverflowError:
        w_ucb = 1.0 if f_best > 5.0 else 0.0
        
    w_ucb = np.clip(w_ucb, 0.0, 1.0)
    
    # Scale UCB to be comparable to EI
    # EI is in objective units. UCB is in objective units.
    # gamma scales the UCB contribution.
    gamma = 0.5
    
    blended = ei + w_ucb * gamma * ucb
    
    # Cost-aware score
    scores = blended / safe_cost
    
    # Final safety checks
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
