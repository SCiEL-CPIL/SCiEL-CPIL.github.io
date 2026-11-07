import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a blend of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with adaptive weights based on exploration-exploitation balance.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure no negative sigma or zero values
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma
    # Handle numerical issues
    Z = np.clip(Z, -10, 10)
    ei = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Calculate LCB (Lower Confidence Bound) for minimization
    # We want to minimize, so LCB = mu - beta * sigma
    # For scoring (higher is better), we return -LCB = -mu + beta * sigma
    # But let's work with the "improvement" perspective
    # A point is attractive if it might be below f_best
    # LCB score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Adaptive beta: start high for exploration, decrease as we find good values
    # Use a schedule based on the ratio of f_best to the typical uncertainty
    sigma_median = np.median(sigma)
    if sigma_median < 1e-9:
        sigma_median = 1e-9
    
    # R = f_best / sigma_median
    # If f_best is large relative to uncertainty, we're early -> explore more (high beta)
    # If f_best is small relative to uncertainty, we're later -> exploit more (low beta)
    R = abs(f_best) / sigma_median
    # beta decays as R decreases
    # beta = beta_max / (1 + R) doesn't work well because R can be huge
    # Let's use: beta = beta_max * exp(-R / R_scale)
    # When R is small (f_best << sigma), beta ~ beta_max (explore)
    # When R is large (f_best >> sigma), beta ~ 0 (exploit)
    # Wait, this is backwards.
    # If f_best is large (bad), we haven't found the min. We should explore.
    # If f_best is small (good), we are close to min. We should exploit.
    # So: Large f_best -> High Beta. Small f_best -> Low Beta.
    # R = f_best / sigma_median.
    # If f_best is large, R is large.
    # We want Beta to increase with R?
    # Or maybe the inverse logic:
    # Early on, f_best is high (e.g., 1.0), sigma is high. R ~ 1.
    # Later, f_best is low (e.g., 0.1), sigma is low. R ~ 1.
    # This ratio isn't quite right.
    
    # Let's look at the standard approach:
    # beta_t = sigma * sqrt(2 * log(t))
    # But we don't have t.
    
    # Alternative: Use the relative improvement potential.
    # If mu < f_best, it's a candidate for improvement.
    # EI handles this naturally.
    
    # Let's try a weighted blend: Score = w * EI + (1-w) * (-LCB_score)
    # where -LCB_score = -mu + beta*sigma
    # Actually, EI is already a good balance.
    # The problem with EI is it can be too greedy.
    # LCB is more exploratory.
    
    # Let's compute a score that is primarily EI but boosted by uncertainty in areas
    # where EI is low but uncertainty is high (potential for discovery).
    
    # Normalized EI:
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_max = 1e-9
    ei_norm = ei / ei_max
    
    # Normalized LCB score (higher is better):
    # We want to maximize -mu + beta*sigma
    # Let's pick a fixed beta for LCB part, or adaptive.
    # Let's use beta = 1.0 for the LCB component to balance exploration.
    lcb_score = -mu + 1.0 * sigma
    lcb_max = np.max(lcb_score)
    lcb_min = np.min(lcb_score)
    if lcb_max - lcb_min < 1e-9:
        lcb_norm = 0.5 * np.ones_like(mu)
    else:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    
    # Blend:
    # Early on (f_best is high), we might want more exploration?
    # Actually, EI tends to explore when f_best is high because (f_best - mu) is large for many points.
    # When f_best is low, EI is only high near the best.
    
    # Let's try a simple adaptive weight on the LCB component.
    # If the best possible EI is small, it means we are in exploitation mode, 
    # but we might be stuck. Adding LCB helps escape.
    # If EI is large, it's fine.
    
    # Let's use a weight that increases as the "confidence" in current best decreases?
    # Or simply:
    # Score = EI + gamma * (sigma - median_sigma)
    # This rewards high uncertainty beyond the median.
    
    sigma_med = np.median(sigma)
    gamma = 0.5  # Exploration boost
    
    # Final Score = EI + gamma * (sigma - sigma_med)
    # This adds a small linear boost to high-uncertainty regions, 
    # encouraging exploration even where EI is low.
    
    scores = ei + gamma * (sigma - sigma_med)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
