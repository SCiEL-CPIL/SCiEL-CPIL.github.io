import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Combines the dynamic budget-aware exploration/exploitation balance from the
    top-performing inspiration with a global cost-budget penalty that shifts
    the balance as the effective budget depletes. Uses a decaying log-cost
    penalty that is strong early (to avoid expensive explorations) and weak
    late (to allow exploitation of cheap high-confidence points).
    """
    t0 = time.time()
    deadline = 320.0  # safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Estimate progress: how close are we to the minimum?
    # Powell 4D on [-4,5]^4 has min 0, typical f_best starts around 10-20
    # Use a robust progress metric based on f_best
    f_range = max(abs(f_best), 1.0) + 1.0
    progress = 1.0 - (f_best / (f_range * 0.5))
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic beta: high exploration early, low exploitation late
    beta = 0.5 + 1.5 * (1.0 - progress)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus scaled by budget-aware beta
    exploration = beta * sigma_safe
    
    # Combine EI and exploration
    # Normalize both to comparable scales
    ei_scale = float(np.std(ei)) + 1e-8
    expl_scale = float(np.std(exploration)) + 1e-8
    
    ei_norm = ei / ei_scale
    expl_norm = exploration / expl_scale
    
    # Weight: shift from exploration to exploitation as budget depletes
    w_ei = 0.3 + 0.7 * progress
    w_expl = 1.0 - w_ei
    
    combined = w_ei * ei_norm + w_expl * expl_norm
    
    # Decaying cost penalty: strong early, weak late
    # Early (f_best high): strong cost penalty to avoid expensive explorations
    # Late (f_best low): weak cost penalty to allow exploitation
    penalty_strength = 0.05 + 0.45 * (1.0 - progress)
    
    # Use log1p for smoothness and boundedness
    cost_penalty = penalty_strength * np.log1p(cost_safe)
    
    score = combined - cost_penalty
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - t0 > deadline - 2.0:
        score = np.where(np.isfinite(score), score, 0.0)
    
    return score
