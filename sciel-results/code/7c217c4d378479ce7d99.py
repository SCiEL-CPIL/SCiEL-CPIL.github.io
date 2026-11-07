"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 2.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a deterministic fallback
    # to avoid hanging in scipy calls or complex logic. 
    # However, scipy.stats.norm.cdf/pdf are vectorized and fast.
    # The main risk is if the caller loops many times. 
    # Since we are just computing scores for 2000 points, it should be fast.
    # But if this function is called in a tight loop, we need to be careful.
    
    if _time_exceeded():
        # Fallback: return cost-aware EI computed with simplified approximation or just EI
        # To be safe and fast, we return a simple EI score.
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Kriging Distance (KD)
    # KD is typically defined as sigma / (distance to nearest observed point).
    # However, we are NOT given the coordinates of observed points.
    # We only have mu, sigma, f_best, cost.
    # Wait, the prompt says: "No coordinates are passed."
    # So we CANNOT compute true Kriging Distance based on physical distance to neighbors.
    # We can only use sigma as a proxy for uncertainty/exploration.
    # Pure EI already balances exploration (via sigma) and exploitation (via f_best - mu).
    # The current seed is EI / cost.
    
    # Let's look at the inspiration 2: "Kriging Distance (KD) ... normalized by distance to nearest observed point".
    # Since we don't have coordinates, we can't do that.
    # However, we can modify the cost-awareness.
    
    # Strategy: Instead of dividing by cost linearly, we can use a different cost model.
    # The cost is 1 + mean(x_scaled). 
    # High x_scaled means high cost.
    # The current seed divides EI by cost.
    
    # Alternative Strategy: Use Upper Confidence Bound (UCB) with cost awareness.
    # UCB = mu + kappa * sigma.
    # We want to maximize UCB / cost? Or maximize (f_best - UCB_min) / cost?
    # Since we are minimizing, we want low mu.
    # EI is generally better for minimization.
    
    # Let's try a different angle: 
    # The cost is 1 + mean(x). 
    # Points with lower x have lower cost.
    # If we assume the function is roughly similar across the domain, 
    # prioritizing low-cost points allows more evaluations.
    
    # Let's try a "Cost-Adjusted EI" where we penalize high cost more aggressively.
    # Current: EI / cost.
    # Alternative: EI / (cost ** alpha) with alpha > 1?
    # Or: EI * (1 / cost) ** 2?
    
    # Let's look at the "Different Strategy" requirement.
    # Current: EI / cost.
    # New: Use a Gaussian Process-based "Probability of Improvement" (PI) or a 
    # hybrid of EI and a bonus for low cost (since low cost allows more samples).
    
    # Actually, a common failure of EI/cost is that it might ignore regions with 
    # high uncertainty if the expected improvement is slightly lower than a 
    # cheap, certain improvement.
    
    # Let's try a "Weighted Sum" approach:
    # Score = w1 * (EI / cost) + w2 * (sigma / cost)
    # This explicitly adds an exploration term (sigma) scaled by cost.
    # This is similar to UCB but scaled by cost.
    
    # Let's define:
    # Exploitation term: EI / cost
    # Exploration term: sigma / cost
    # We need to balance them. 
    # Standard EI includes sigma. 
    # If we add extra sigma, we increase exploration.
    
    # Let's try: Score = (EI + beta * sigma) / cost
    # With beta = 1.0?
    
    # Another idea: The cost function is 1 + mean(x). 
    # In the scaled [0,1] space, the minimum cost is 1 (at x=0) and max is 2 (at x=1).
    # The variation in cost is small (factor of 2).
    # EI varies much more.
    # So cost-awareness might be secondary.
    
    # What if we use a "Cost-Weighted UCB"?
    # UCB = mu + kappa * sigma (for minimization, we want to minimize this, 
    # but acquisition functions usually maximize a score).
    # For minimization, a common acquisition is -UCB or EI.
    # Let's stick to EI but modify the cost division.
    
    # Let's try a strategy that prioritizes "cheap" points more strongly 
    # to allow more total evaluations before the budget runs out.
    # Score = EI / (cost ** 2)
    
    # Let's implement Score = EI / (cost ** 2)
    # This is a materially different cost model (quadratic penalty vs linear).
    
    # To ensure we don't divide by zero or have issues:
    cost_safe = np.maximum(cost, 1.0) # Cost is always >= 1
    
    score = ei / (cost_safe ** 2)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
