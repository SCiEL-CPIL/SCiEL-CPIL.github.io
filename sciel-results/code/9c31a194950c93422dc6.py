import numpy as np
from scipy.stats import norm
import time

# Deadline guard constants
START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Blended acquisition: Cost-normalized EI + UCB exploration term.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Compute a UCB-like term: -mu + kappa * sigma.
    3. Blend the two: score = (EI + alpha * UCB) / cost.
    4. Use adaptive alpha and kappa based on f_best to balance exploration/exploitation.
    
    The EI term provides principled probabilistic improvement.
    The UCB term ensures exploration in high-uncertainty regions where EI is flat/zero.
    Cost normalization accounts for the cost-aware setting.
    """
    # Check time budget
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)
    
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
    # alpha controls how much UCB contributes relative to EI
    # Early on (high f_best): EI is large, UCB also large. We want both.
    # Late on (low f_best): EI is small, UCB may dominate if kappa is not small.
    # We want EI to dominate late, UCB to help early.
    # alpha = 0.5 + 0.5 * sigmoid(-(f_best - 5))
    # If f_best = 10: sigmoid(-5) ~ 0, alpha ~ 0.5
    # If f_best = 0: sigmoid(5) ~ 1, alpha ~ 1.0? No, we want alpha low when f_best is low.
    # Let's reverse: we want UCB weight HIGH when f_best is HIGH.
    # w_ucb = 1 / (1 + exp((f_best - 5) * 0.5))
    # If f_best = 10: exp(2.5) ~ 12, w ~ 1/13 ~ 0.08 (Low? We want HIGH early)
    # Wait, "early" means f_best is HIGH. So we want w_ucb HIGH when f_best is HIGH.
    # Let's use: w_ucb = 1 / (1 + exp((5 - f_best) * 0.5))
    # If f_best = 10: exp(-2.5) ~ 0.08, w ~ 0.92 (High)
    # If f_best = 0: exp(2.5) ~ 12, w ~ 0.08 (Low)
    # This seems right. High UCB weight early, low late.
    
    try:
        w_ucb = 1.0 / (1.0 + np.exp((5.0 - f_best) * 0.5))
    except OverflowError:
        w_ucb = 1.0 if f_best > 5.0 else 0.0
        
    w_ucb = np.clip(w_ucb, 0.0, 1.0)
    
    # Scale UCB to be comparable to EI
    # EI is in objective units. UCB is in objective units.
    # But EI is bounded by f_best, while UCB can be negative/positive.
    # Let's blend: score_raw = ei + w_ucb * gamma * ucb
    # What is gamma? 
    # If mu ~ f_best, EI ~ sigma * phi(0) ~ 0.4*sigma. UCB ~ sigma*kappa.
    # We want them to be on similar scales.
    # Let's use gamma = 0.5.
    
    gamma = 0.5
    blended = ei + w_ucb * gamma * ucb
    
    # Ensure blended is non-negative? Not strictly necessary, but helps with ranking.
    # If both EI and UCB are negative, the point is bad.
    # Let's allow negative values but floor them slightly below 0 to avoid all-zero ties.
    # Actually, just use the value. Higher is better.
    
    # Cost-aware score
    scores = blended / safe_cost
    
    # Final safety checks
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
