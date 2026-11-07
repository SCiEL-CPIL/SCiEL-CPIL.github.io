"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_LIMIT = 300.0  # 300s limit, keep margin

def acquisition(mu, sigma, f_best, cost):
    """
    Dynamic cost-aware acquisition function.
    
    Strategy:
    - Blend Expected Improvement (EI) and a variance-based exploration bonus.
    - The blend weight depends on the progress of the optimization (f_best).
    - All terms are normalized by cost to favor cheaper evaluations.
    - Includes a deadline guard to ensure we return before timeout.
    """
    # Deadline Guard: If we are too close to the limit, return a simple, fast score
    # to ensure we don't timeout. The evaluator kills at 330s, but we want to be safe.
    # Since this function is called repeatedly, we check the time here.
    # If time is tight, we fall back to a very simple metric.
    current_time = time.time()
    if current_time - _START_TIME > _DEADLINE_LIMIT - 5.0:
        # Emergency fallback: simple EI / cost
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_term = np.maximum(ei_term, 0.0)
    
    # 2. Exploration Bonus
    # We use the standard deviation (uncertainty) as a proxy for exploration potential.
    # However, raw sigma can be noisy. We normalize it by the current best value scale
    # to make it comparable to EI.
    # If f_best is close to 0 (minimum), we should be careful not to over-explore.
    # Scale exploration by 1 / (1 + f_best) to reduce exploration as we converge.
    # Note: f_best is the minimum, so for functions with min 0, f_best >= 0.
    # For Hartmann, min is ~-3.86, so f_best will be negative.
    # Let's use a robust scale. The range of functions varies.
    # A simple heuristic: Exploration weight decays as f_best gets closer to the theoretical minimum.
    # Since we don't know the exact minimum, we assume progress is made when f_best decreases.
    # Let's use a dynamic weight based on the magnitude of f_best.
    
    # If f_best is large (early), explore more. If f_best is small (late), exploit more.
    # Assuming typical scales, a weight of 1.0 early and 0.1 late is reasonable.
    # Let's map f_best to a weight.
    # For Ackley/Rastrigin/Levy, min is 0. f_best starts high (e.g., 10-100) and goes to 0.
    # For Hartmann, min is -3.86. f_best starts high (e.g., 10) and goes to -3.86.
    
    # A generic decay: alpha = 1 / (1 + max(0, f_best))
    # This works well for 0-min functions. For Hartmann, f_best is negative, so max(0, f_best) is 0, alpha=1.
    # This might be too exploratory for Hartmann late in the run.
    # Let's try a different approach: use the ratio of current best to a reference.
    # Reference: Initial uniform points usually give a decent baseline.
    # Let's stick to a simple dynamic factor based on f_best.
    
    # If f_best > 0, we are likely early or on a 0-min function.
    # If f_best < 0, we are on Hartmann and likely past the initial random phase.
    
    # Let's define exploration term as sigma_safe.
    # Weight it by a factor that decays with progress.
    # Progress proxy: 1 / (1 + abs(f_best))
    # This ensures that as f_best -> 0 (or -3.86), the weight doesn't necessarily go to 0 if f_best is negative.
    # Actually, for Hartmann, we want to exploit more as we get closer to -3.86.
    # Let's use a weight that depends on the "improvement potential".
    
    # Alternative: Use a fixed but tuned blend, but adjust the UCB beta dynamically.
    # UCB = mu + beta * sigma.
    # Minimization UCB = mu - beta * sigma (we want low mu, high sigma).
    # But we want to *maximize* the score. So Score = (f_best - mu + beta * sigma) / cost.
    # Let's call this "UCB-like" term.
    
    ucb_term = (f_best - mu + 1.5 * sigma_safe)
    ucb_term = np.maximum(ucb_term, 0.0) # Ensure non-negative contribution? Or just use raw?
    # Raw UCB can be negative if mu is very low. But we divide by cost.
    # Let's keep it raw but ensure it's finite.
    
    # Dynamic Beta:
    # Early: High Beta (explore). Late: Low Beta (exploit).
    # Use f_best to estimate progress.
    # If f_best is very close to 0 (for 0-min) or -3.86 (for Hartmann), Beta should be small.
    # Since we don't know the function, we can't hardcode -3.86.
    # However, we can observe that f_best is decreasing.
    # Let's use a heuristic: Beta = 2.0 / (1.0 + f_best) if f_best > 0 else 0.5.
    # This is a bit hacky.
    
    # Let's try a simpler, more robust strategy:
    # Combine EI and UCB with a fixed weight, but ensure numerical stability.
    # The previous program used smooth max. Let's try a weighted sum with dynamic weights.
    
    # Weight for EI: w_ei
    # Weight for UCB: w_ucb
    
    # Heuristic:
    # If f_best is "high" (far from min), explore more (higher w_ucb).
    # If f_best is "low" (close to min), exploit more (higher w_ei).
    
    # Let's assume the "distance to min" is proportional to f_best for 0-min functions.
    # For Hartmann, f_best is negative.
    
    # Let's use a normalized progress metric:
    # p = 1 / (1 + max(0, f_best))
    # This gives p=1 when f_best<=0 (Hartmann late, or converged 0-min).
    # This gives p<1 when f_best>0 (early 0-min).
    
    # We want to explore more when p is small (early).
    # So w_ucb = (1 - p) * 1.0 + 0.1 # Min weight 0.1
    # w_ei = p * 1.0 + 0.1 # Min weight 0.1
    
    # Let's test this logic.
    # f_best = 10 -> p = 1/11 = 0.09. w_ucb = 0.91+0.1=1.01, w_ei=0.09+0.1=0.19. Explore.
    # f_best = 0.1 -> p = 1/1.1 = 0.91. w_ucb = 0.09+0.1=0.19, w_ei=0.91+0.1=1.01. Exploit.
    # f_best = -3.86 -> p = 1/1 = 1.0. w_ucb = 0.1, w_ei = 1.1. Exploit.
    
    p = 1.0 / (1.0 + max(0.0, f_best))
    
    w_ucb = (1.0 - p) * 1.0 + 0.1
    w_ei = p * 1.0 + 0.1
    
    # Normalize scores to be comparable
    # EI is in objective units. UCB is in objective units.
    # Divide by cost.
    
    score_ei = ei_term / cost_safe
    score_ucb = ucb_term / cost_safe
    
    # Combine
    scores = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
