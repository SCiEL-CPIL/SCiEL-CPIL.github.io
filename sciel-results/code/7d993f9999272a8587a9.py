# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) for exploitation
    2. UCB for exploration
    3. Adaptive weighting based on progress
    
    For minimization:
    - EI = E[f_best - f(x)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    - UCB = mu - beta * sigma (we want low values)
    
    We maximize a weighted combination.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # f_best is the lowest observed value
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score (we want to maximize, so we negate the lower bound)
    # UCB for minimization: we prefer low mu and high sigma
    # Let's use a beta that adapts to progress
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # f_best ranges from ~1.0 (early) to ~0.0 (late)
    f_norm = np.clip(f_best, 0.0, 1.0)
    
    # Beta decreases as we get closer to optimum
    beta = 0.5 + 2.0 * f_norm  # beta in [0.5, 2.5]
    
    # UCB score = beta * sigma - mu (maximize this)
    ucb_score = beta * sigma - mu
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max if ucb_max > 0 else 0.0
    
    # Adaptive weighting: more EI when close to optimum, more UCB when far
    # weight_ei increases as f_best decreases
    weight_ei = 0.3 + 0.7 * (1.0 - f_norm)  # 0.3 to 1.0
    weight_ucb = 1.0 - weight_ei * 0.5  # 0.5 to 0.85
    
    # Combine
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
