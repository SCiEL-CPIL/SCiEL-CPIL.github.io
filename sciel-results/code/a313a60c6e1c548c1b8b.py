import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending LCB and EI with adaptive exploration.
    """
    t_start = time.time()
    deadline = t_start + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety
    sigma = np.clip(sigma, 1e-9, None)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    cost = np.maximum(cost, 1e-6)

    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive Beta for LCB
    # Ackley max is ~29.
    # We want beta to be high when f_best is high (early search)
    # and low when f_best is low (late search).
    f_scale = 25.0
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    
    # Beta range: 
    # Early: explore more (beta ~ 3.0)
    # Late: exploit more (beta ~ 0.5)
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio

    # LCB for minimization: we want to maximize the score
    # Score = -mu + beta * sigma
    lcb_score = -mu + beta * sigma

    # Blend LCB and EI
    # In early stages (high f_ratio), LCB dominates.
    # In late stages (low f_ratio), EI dominates.
    # We can use f_ratio as a weight, but let's try a fixed blend or dynamic.
    # Let's try: score = (1 - f_ratio) * EI + f_ratio * LCB
    # This ensures that when f_best is high (f_ratio ~ 1), we use LCB.
    # When f_best is low (f_ratio ~ 0), we use EI.
    
    blend_weight = f_ratio
    blended_score = (1.0 - blend_weight) * ei + blend_weight * lcb_score

    # Cost-aware normalization
    # Use a softer penalty than 1/cost to allow some exploration of expensive points if they have high uncertainty.
    # cost^0.5 was mentioned in lessons as improving score.
    cost_weight = 1.0 / np.sqrt(cost)
    
    score = blended_score * cost_weight
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > deadline:
        return score

    return score
