# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return inverse cost (cheapest)
        return 1.0 / np.maximum(cost, 1e-10)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Cost-Weighted Score
    # We want to maximize EI per unit cost.
    # Score = EI / Cost
    # This directly optimizes for the expected reduction in regret per unit of cost.
    
    cost_safe = np.maximum(cost, 1e-10)
    score = ei / cost_safe
    
    # 3. Fallback and Safety
    # If all scores are 0 (or very close), it means we are confident no point improves f_best.
    # In this case, we should still pick a point to continue the search, preferably one
    # that might reveal new information or is cheap.
    # If EI is 0, it implies f_best - mu <= 0 and sigma is small, or z is very negative.
    # Actually, if EI is 0, the GP thinks f(x) >= f_best.
    # We should explore areas of high uncertainty (high sigma) or low cost.
    # Let's add a small exploration bonus based on sigma, weighted by 1/cost.
    # This ensures we don't get stuck if EI is 0 everywhere.
    
    # Exploration bonus: favor high sigma (uncertainty) and low cost.
    # Normalize sigma to [0, 1] for the bonus?
    # Or just use raw sigma?
    # Let's use a small weight.
    exploration_bonus = sigma_safe / cost_safe
    exploration_weight = 1e-4
    
    # Only add bonus if the max EI score is very low, to not disturb the main signal.
    max_ei_score = np.max(score)
    if max_ei_score < 1e-6:
        score = exploration_bonus
    else:
        # Add a tiny bit of exploration to break ties or handle numerical zeros
        score = score + exploration_weight * exploration_bonus

    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

# EVOLVE-BLOCK-END
