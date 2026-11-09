"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Internal state for time-based adaptation
_start_time = None
_total_budget = 330.0  # Seconds
_safety_margin = 10.0  # Seconds to return before kill

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are running out of time, return a safe, simple score
    # to ensure the function returns quickly and doesn't crash the evaluator.
    elapsed = time.time() - _start_time
    if elapsed > _total_budget - _safety_margin:
        # Fallback: simple EI
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost + 1e-9)

    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Uncertainty (U) - standard deviation
    u = sigma
    
    # 3. Normalize EI and U to [0, 1] range for fair combination
    # We use min-max normalization on the current batch of candidates
    ei_min, ei_max = np.min(ei), np.max(ei)
    u_min, u_max = np.min(u), np.max(u)
    
    ei_range = ei_max - ei_min
    u_range = u_max - u_min
    
    if ei_range < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    if u_range < 1e-9:
        u_norm = np.zeros_like(u)
    else:
        u_norm = (u - u_min) / u_range

    # 4. Adaptive Weighting based on Convergence State
    # As we spend more cost, we should prioritize exploitation (EI) over exploration (U)
    # f_best is the current best observed value.
    # A simple proxy for convergence is the ratio of f_best to the initial scale.
    # However, we don't know the initial scale directly. 
    # Instead, we can use the cost spent so far as a proxy for progress.
    # The task says "While total cost is below 30".
    # Let's estimate progress based on the magnitude of f_best relative to a typical scale.
    # For Three Hump Camel, values are small. 
    # A more robust approach: use the ratio of current cost to max cost (30).
    # But we don't have the total cost spent, only the cost vector for the current batch.
    # The cost vector 'cost' has shape (2000,). 
    # Wait, the prompt says "cost have shape (2000,)". 
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # The cost passed to acquisition is likely the cost of the *candidates*? 
    # Or is it the cost spent so far? 
    # "mu, sigma and cost have shape (2000,). Return 2000 finite scores"
    # "Runs start with 5 uniform random points (cost counted). While total cost is below 30, the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    # This implies 'cost' is the cost associated with each candidate x in the batch of 2000.
    # We don't explicitly receive the *total* cost spent so far. 
    # However, we can infer the "stage" of the optimization from f_best.
    # If f_best is close to 0, we are converged. 
    # Let's use a heuristic: 
    # weight_exploration = max(0, 1 - f_best / 10)  # Arbitrary scale
    # This is risky. 
    
    # Alternative: Use the distribution of 'cost' in the batch.
    # Candidates with lower cost are generally "cheaper" to evaluate.
    # The baseline EI/cost already handles this.
    # The structural change should be in the *combination* of EI and Uncertainty.
    
    # Let's use a fixed adaptive weight based on the number of evaluations inferred from f_best?
    # No, let's use a simpler structural change: 
    # **Power-Law Cost Penalty with Adaptive Exponent**
    # Instead of linear division by cost, use a penalty that becomes stronger as f_best improves.
    
    # Actually, the most robust structural change without knowing total time spent is:
    # **Probability of Improvement (PI) blended with EI**
    # PI is more robust to noise and scale than EI.
    # pi = norm.cdf(z)
    
    # Let's try a **Weighted Sum of Normalized EI and PI** with a cost penalty.
    # score = (w1 * ei_norm + w2 * pi_norm) / cost^alpha
    
    # To make it adaptive, we can estimate the "remaining budget" implicitly.
    # If f_best is very small, we are likely late in the process.
    # Let's assume a typical range for Three Hump Camel is [-1.5, 0].
    # If f_best < -1.0, we are deep in exploitation.
    
    # Let's stick to the **Multi-objective Pareto** idea from the inspiration that scored 1.599.
    # It used "adaptive exploitation/exploration weighting based on convergence state".
    
    # Convergence State Proxy: 
    # Let's use the relative improvement of the best candidate in the batch vs f_best.
    # Or simply use a fixed schedule based on the magnitude of f_best.
    
    # If f_best is close to the global min (0), we want high exploitation.
    # If f_best is far, we want high exploration.
    
    # Scale f_best to [0, 1] for weighting.
    # Assume max possible f_best is around 5 (arbitrary upper bound for this function).
    # f_best is negative.
    # progress = 1 - (f_best / 5.0)  # If f_best=-5, progress=2. If f_best=0, progress=1.
    # This is not normalized.
    
    # Let's use a sigmoid function on f_best to create a smooth transition.
    # weight_exploit = 1 / (1 + np.exp(-10 * f_best))
    # If f_best = -1, weight_exploit ~ 0.00004 (very low) -> High Exploration?
    # Wait, if f_best is bad (large negative), we want to explore.
    # If f_best is good (close to 0), we want to exploit.
    
    # Let's invert:
    # f_best is in [-1.5, 0].
    # Let's map f_best to [0, 1] where 0 is bad, 1 is good.
    # f_best_norm = (f_best + 2.0) / 2.0  # Maps [-2, 0] to [0, 1]
    # Clip to [0, 1]
    f_best_norm = np.clip((f_best + 2.0) / 2.0, 0.0, 1.0)
    
    # Weight for Exploitation (EI) increases as f_best improves (gets closer to 0)
    # Weight for Exploration (U) decreases as f_best improves
    w_exploit = f_best_norm
    w_explore = 1.0 - f_best_norm
    
    # To ensure we don't completely drop one, add a small epsilon
    w_exploit = np.maximum(w_exploit, 0.1)
    w_explore = np.maximum(w_explore, 0.1)
    
    # Normalize weights
    w_sum = w_exploit + w_explore
    w_exploit /= w_sum
    w_explore /= w_sum
    
    # Combined Score
    # We use the normalized EI and U
    combined = w_exploit * ei_norm + w_explore * u_norm
    
    # Apply Cost Penalty
    # Use a power law with alpha = 1.0 (linear) to match baseline behavior but on the combined score
    # Or alpha < 1.0 to be less penalizing?
    # Let's stick to linear division for stability, but the adaptive weighting is the key change.
    
    scores = combined / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
