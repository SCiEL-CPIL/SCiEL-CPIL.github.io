"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safe fallback: return negative mu (exploitation)
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Structural change: Use a hybrid of Expected Improvement and UCB
    # For minimization, we want to find points that are likely to have low values
    # and have high uncertainty.
    
    # Compute EI (Expected Improvement) for minimization
    # EI = E[max(0, f_best - f(x))]
    # If f(x) ~ N(mu, sigma^2), then:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Standard EI for minimization
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Handle numerical issues: where sigma is very small, EI should be close to max(0, f_best - mu)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # For minimization, we want low values, so we use LCB = mu - beta * sigma
    # We want to maximize the score, so we return -LCB = -mu + beta * sigma
    # But we want to blend this with EI.
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # Total 25 acquisition calls (30 evals - 5 initial)
    if _call_count <= 3:
        beta = 3.0
        weight_ei = 0.3
    elif _call_count <= 6:
        beta = 2.0
        weight_ei = 0.5
    elif _call_count <= 10:
        beta = 1.5
        weight_ei = 0.7
    elif _call_count <= 15:
        beta = 1.0
        weight_ei = 0.85
    elif _call_count <= 20:
        beta = 0.7
        weight_ei = 0.95
    else:
        beta = 0.5
        weight_ei = 1.0
    
    # UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    # Normalize EI and UCB scores to [0, 1] for blending
    # First, handle edge cases
    ei_min = np.min(ei) if len(ei) > 0 else 0
    ei_max = np.max(ei) if len(ei) > 0 else 1
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb_score) if len(ucb_score) > 0 else 0
    ucb_max = np.max(ucb_score) if len(ucb_score) > 0 else 1
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Blend the two scores
    scores = weight_ei * ei_norm + (1 - weight_ei) * ucb_norm
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
