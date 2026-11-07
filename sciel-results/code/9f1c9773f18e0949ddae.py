import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition for minimizing the objective.
    
    The exploration coefficient beta is adapted based on the current state of the optimization:
    beta = max(0.5, (f_best - min(mu)) / (max(sigma) + eps))
    
    This encourages exploration when the gap between the best observed value and the best
    predicted value is large relative to the uncertainty, and shifts to exploitation as
    the GP posterior tightens around the optimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = f_best - mu
        return np.clip(score, -1e12, 1e12)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Calculate adaptive beta
    # If max(sigma) is very small, beta should be small (exploit).
    # If f_best is much larger than min(mu), beta should be large (explore).
    mu_min = np.min(mu)
    sigma_max = np.max(sigma_safe)
    
    # Avoid division by zero
    if sigma_max < eps:
        beta = 0.5
    else:
        # Scale the gap by the uncertainty
        # If f_best == mu_min, gap is 0, beta = 0.5 (minimum exploration)
        # If f_best >> mu_min, beta grows.
        gap = f_best - mu_min
        beta = max(0.5, gap / sigma_max)
        
        # Cap beta to prevent extreme exploration in very noisy/early stages
        # A beta of 10 is very exploratory. 1.0 is moderate.
        # Let's cap at 5.0 to be safe.
        beta = min(beta, 5.0)

    # LCB Score: We want to MINIMIZE f(x).
    # Lower bound on f(x) is mu(x) - beta * sigma(x).
    # We want the point with the LOWEST lower bound.
    # Since the interface requires HIGHER score to be better, we negate the LCB value
    # or use (f_best - LCB).
    # Let's use: Score = (f_best - mu) - beta * sigma
    # This is high when mu is low (good) and sigma is low (certain), 
    # BUT wait: -beta * sigma means high sigma DECREASES the score.
    # This is EXPLOITATION.
    
    # LCB for minimization:
    # We want to find x such that f(x) is small.
    # The lower confidence bound is LB(x) = mu(x) - beta * sigma(x).
    # We want to MINIMIZE LB(x).
    # So the "best" point has the MINIMUM LB(x).
    # The acquisition function usually returns a value to MAXIMIZE.
    # So we should return -LB(x) = -mu(x) + beta * sigma(x).
    # OR, we can return (f_best - LB(x)) = f_best - mu(x) + beta * sigma(x).
    
    # Let's check the current EI logic:
    # EI = (f_best - mu) * Phi + sigma * phi.
    # If sigma is large, EI is large (exploration).
    
    # For LCB:
    # If we return Score = -mu + beta * sigma:
    # High sigma -> High Score (Exploration).
    # Low mu -> High Score (Exploitation).
    # This matches the behavior we want.
    
    score = -mu + beta * sigma_safe

    # Keep the output strictly finite and bounded.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.clip(score, -1e12, 1e12)

    # Guarantee exactly `n` finite entries (defensive).
    if score.shape[0] != n:
        score = np.resize(score, n)

    return score
