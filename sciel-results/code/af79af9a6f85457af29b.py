# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Fast CDF/PDF approximations to speed up inner loops
def norm_cdf_fast(z):
    """Approximate normal CDF using Abramowitz and Stegun approximation (fast, accurate)"""
    # Handle NaN/Inf
    z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    d = 0.3989422804014327 * np.exp(-z * z / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    out = 1.0 - p
    out = np.where(z < 0, 1.0 - out, out)
    return out

def norm_pdf_fast(z):
    """Approximate normal PDF"""
    z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    return 0.3989422804014327 * np.exp(-0.5 * z * z)

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Hybrid Acquisition Function for Bayesian Optimization (Rosenbrock 2D).
    
    Strategy:
    1. Compute both Expected Improvement (EI) and Upper Confidence Bound (UCB) scores.
    2. Normalize both by cost to create cost-aware efficiency scores.
    3. Dynamically blend between UCB and EI based on the current best value (f_best).
       - When f_best is large (poor), use UCB to explore broadly and find better basins.
       - When f_best is small (good), use EI to exploit the local neighborhood of the minimum.
    4. The transition is smooth and based on a logistic function of f_best.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    scores: (2000,) array of finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_raw = (f_best - mu) * norm_cdf_fast(z) + sigma * norm_pdf_fast(z)
    ei_raw = np.maximum(ei_raw, 0.0)
    ei_raw = np.where(np.isfinite(ei_raw), ei_raw, 0.0)
    
    # Calculate UCB for minimization
    # We want to maximize the score, so we define the score as -mu + beta * sigma
    # Standard UCB for minimization often uses a fixed beta. 
    # Here we use a beta that might be scaled, but let's stick to a robust beta first.
    # Beta = 1.0 is a common starting point. 
    ucb_raw = -mu + 1.5 * sigma  # Slightly aggressive exploration
    
    # Normalize raw scores to [0, 1] range to make them comparable before blending?
    # Normalizing can be dangerous if the range is tiny. 
    # Instead, let's normalize by the max value in the batch to scale them similarly.
    # This ensures that if EI values are generally smaller than UCB values (or vice versa),
    # they contribute similarly to the blend.
    
    max_ei = np.max(ei_raw)
    max_ucb = np.max(ucb_raw)
    
    # Avoid division by zero
    if max_ei < 1e-9:
        ei_norm = np.zeros_like(ei_raw)
    else:
        ei_norm = ei_raw / max_ei
        
    if max_ucb < 1e-9:
        ucb_norm = np.zeros_like(ucb_raw)
    else:
        ucb_norm = ucb_raw / max_ucb
        
    # Cost-aware scores
    score_ei = ei_norm / cost
    score_ucb = ucb_norm / cost
    
    # Determine blend weight based on f_best
    # If f_best is high (e.g., > 10), we want more UCB (w_ucb -> 1).
    # If f_best is low (e.g., < 1), we want more EI (w_ei -> 1).
    # Use a logistic function: w_ucb = 1 / (1 + exp(k * (f_best - center)))
    # Let's say center = 5.0, k = 0.5
    # If f_best = 5, w_ucb = 0.5
    # If f_best = 10, w_ucb = 1 / (1 + exp(2.5)) ~ 0.076 (Wait, I want high f_best -> high UCB)
    # So I need: w_ucb = 1 / (1 + exp(-k * (f_best - center)))
    # If f_best = 10, exp(2.5) is large, w_ucb -> 1.
    # If f_best = 0, exp(-2.5) is small, w_ucb -> 0.
    
    center = 5.0
    k = 0.5
    # Clip f_best to avoid overflow in exp
    f_best_clipped = np.clip(f_best, -100, 100)
    w_ucb = 1.0 / (1.0 + np.exp(-k * (f_best_clipped - center)))
    w_ei = 1.0 - w_ucb
    
    # Blend the scores
    score = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
