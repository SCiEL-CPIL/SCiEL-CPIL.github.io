"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining PI/cost ranking with adaptive
    cost exponent and mild EI exploitation.
    
    Primary signal: rank-normalized Probability of Improvement per unit cost
    with adaptive cost exponent that sharpens penalty near convergence.
    Refinement: small EI component to exploit regions where mean < f_best.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma and cost
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    pi = np.clip(pi, 0.0, 1.0)
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost exponent: as f_best approaches 0, increase penalty for high cost
    # This helps navigate the Rosenbrock valley by staying in cheaper areas near convergence
    f_best_abs = abs(f_best)
    if f_best_abs < 0.01:
        cost_exponent = 1.8
    elif f_best_abs < 1.0:
        # Interpolate from 1.0 to 1.8 as f_best goes from 1.0 to 0.01
        cost_exponent = 1.0 + 0.8 * (1.0 - f_best_abs) / 0.99
    else:
        cost_exponent = 1.0
    
    # Apply adaptive cost penalty
    cost_adj = cost_safe ** cost_exponent
    
    # PI per unit cost
    pi_cost = pi / cost_adj
    
    # EI per unit cost (small weight)
    ei_cost = ei / cost_adj
    
    # Rank-based scoring for PI/cost (primary signal)
    ranks_pi = np.argsort(np.argsort(pi_cost))
    n = len(pi_cost)
    score_pi_rank = ranks_pi.astype(float) / max(n - 1, 1)
    
    # Rank-based scoring for EI/cost (secondary signal)
    ranks_ei = np.argsort(np.argsort(ei_cost))
    score_ei_rank = ranks_ei.astype(float) / max(n - 1, 1)
    
    # Normalize PI/cost magnitude
    pi_cost_min = np.min(pi_cost)
    pi_cost_max = np.max(pi_cost)
    if pi_cost_max - pi_cost_min > 1e-12:
        pi_cost_norm = (pi_cost - pi_cost_min) / (pi_cost_max - pi_cost_min)
    else:
        pi_cost_norm = np.zeros_like(pi_cost)
    
    # Normalize EI/cost magnitude
    ei_cost_min = np.min(ei_cost)
    ei_cost_max = np.max(ei_cost)
    if ei_cost_max - ei_cost_min > 1e-12:
        ei_cost_norm = (ei_cost - ei_cost_min) / (ei_cost_max - ei_cost_min)
    else:
        ei_cost_norm = np.zeros_like(ei_cost)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        scores = 0.7 * score_pi_rank + 0.3 * pi_cost_norm
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Blend: primarily PI/cost ranking, with mild EI contribution
    # PI/cost is the main signal; EI adds exploitation for regions where mean < f_best
    # Weight PI more heavily since it proved superior (2.743 vs 1.93 for EI-based)
    w_pi_rank = 0.45
    w_pi_mag = 0.25
    w_ei_rank = 0.15
    w_ei_mag = 0.15
    
    scores = (w_pi_rank * score_pi_rank + 
              w_pi_mag * pi_cost_norm + 
              w_ei_rank * score_ei_rank + 
              w_ei_mag * ei_cost_norm)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 290:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
