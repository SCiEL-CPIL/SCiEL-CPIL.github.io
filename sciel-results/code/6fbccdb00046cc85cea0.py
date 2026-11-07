"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s safety margin within 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Dynamic Beta LCB with cost power penalty.
    
    Structural Change:
    - Replaces fixed-weight EI/Mu blend with a Dynamic Lower Confidence Bound (LCB) 
      based improvement metric.
    - Uses a power-law cost penalty (cost^gamma) instead of linear division to 
      more aggressively avoid expensive regions when uncertainty is low.
    - Beta parameter for exploration decays dynamically as the budget depletes.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe, finite fallback: rank by lowest mu (exploitation)
        # Invert mu so higher score = lower mu
        return -mu.copy()
    
    # Ensure arrays are float64 and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive and clip to a reasonable range to avoid extreme penalties
    # Cost is in [1, 2] typically (1 + mean(x_scaled) where x_scaled in [0,1])
    cost_safe = np.clip(cost, 1.0, 2.0)
    
    # 1. Dynamic Beta Parameter
    # As we spend more budget, we should exploit more (lower beta).
    # f_best is the current best value. The theoretical min is -0.8.
    # We can use the "progress" towards the minimum to adjust beta.
    # Alternatively, use the number of evaluations or total cost spent.
    # Since we don't have total cost directly, we can infer it from the spread of mu/sigma 
    # or just use a fixed schedule if we had iteration count. 
    # However, f_best gives a hint. If f_best is close to -0.8, exploit.
    # Let's use a heuristic based on f_best.
    # The range of cosine8 is roughly [-0.8, 0.8].
    # progress = (f_best - (-0.8)) / (0.8 - (-0.8)) = (f_best + 0.8) / 1.6
    # If f_best is -0.8, progress = 0 (exploit). If f_best is 0.8, progress = 1 (explore).
    # But f_best starts high and decreases.
    # Let's define "remaining potential" = f_best - (-0.8).
    # If remaining potential is large, we need to explore (high beta).
    # If small, exploit (low beta).
    
    min_possible = -0.8
    # Clamp f_best to be above min_possible to avoid negative potential
    potential = np.maximum(f_best - min_possible, 1e-6)
    
    # Normalize potential. Max potential is roughly 1.6 (from -0.8 to 0.8)
    # Let's assume max potential is 1.0 for scaling purposes, or just use it directly.
    # Beta ranges from 0.5 (exploit) to 2.5 (explore).
    # beta = 0.5 + 2.0 * (potential / max_potential)
    # If potential is high, beta is high.
    max_potential = 1.5  # Approximate range of the function
    normalized_potential = np.clip(potential / max_potential, 0.0, 1.0)
    beta = 0.5 + 2.0 * normalized_potential
    
    # 2. Compute LCB-based Improvement
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize f, so a "good" LCB is low.
    # Improvement potential = f_best - LCB(x)
    # We only care if LCB < f_best, otherwise improvement is 0.
    lcb = mu - beta * sigma_safe
    improvement = np.maximum(f_best - lcb, 0.0)
    
    # 3. Cost-Aware Score
    # Use a power penalty on cost.
    # Cost is in [1, 2].
    # If we use cost^1, it's linear. If cost^2, it's quadratic.
    # Let's try gamma = 1.5 for a strong penalty on expensive points.
    gamma = 1.5
    cost_penalty = np.power(cost_safe, gamma)
    
    # Score = Improvement / Cost_Penalty
    # This maximizes "expected improvement per unit of cost^gamma"
    score = improvement / cost_penalty
    
    # 4. Numerical Safety
    # If improvement is 0 everywhere, score is 0. 
    # We should still prefer lower mu if improvement is tied at 0?
    # The problem states "higher = evaluate next".
    # If all scores are 0, we return 0. The evaluator picks one? 
    # Usually, we want to break ties.
    # Let's add a small exploitation term for ties or when improvement is low.
    # Exploitation term: -mu (lower mu is better).
    # Normalize -mu to [0, 1] to blend.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-8)
    mu_norm = (mu_max - mu) / mu_range  # 1.0 for best mu
    
    # If the max improvement score is very small, blend in mu to ensure progress
    max_score = np.max(score)
    if max_score < 1e-6:
        # Pure exploitation mode
        score = mu_norm
    else:
        # Blend: primarily cost-adjusted improvement, but if improvement is low, 
        # the relative ranking by mu might help.
        # Actually, if improvement > 0, it dominates.
        # Let's just ensure finite values.
        pass
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Tie-breaking
    indices = np.arange(len(score), dtype=np.float64)
    score += indices * 1e-10
    
    return score
# EVOLVE-BLOCK-END
