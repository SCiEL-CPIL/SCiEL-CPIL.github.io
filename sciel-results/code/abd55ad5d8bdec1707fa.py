import numpy as np
from scipy.stats import norm
import time

# Global state to track time if needed, though single call is fast
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PoI) with exploration.
    
    Args:
        mu: array of shape (n,) - posterior mean
        sigma: array of shape (n,) - posterior standard deviation
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (n,) - acquisition scores
    """
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Since we minimize, f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Exploration term: 
    # We want to explore areas with high sigma, but only if they have some chance of improvement.
    # A common heuristic is to add a small weight to sigma, or use UCB-like logic.
    # However, the directive suggests a structural change. 
    # Let's try a "Thompson Sampling" like score or a modified PoI.
    # Actually, the inspiration mentions "Multi-Objective Expected Improvement" but 
    # the cell finding says "Adding a small exploration bonus to the probability-based 
    # acquisition function may help escape local minima".
    
    # Let's implement a score that is primarily PoI but boosted by uncertainty.
    # Score = PoI + alpha * sigma / (1 + sigma)  # Bounded exploration
    # Or simpler: Score = PoI + beta * sigma
    
    # Let's look at the "MOEI" directive again: "treats the posterior mean and the 
    # lower confidence bound (LCB with beta=0) as two separate objectives".
    # This is a bit abstract for a single return value. 
    # However, a very strong structural alternative to EI is **Probability of Improvement (PoI)**.
    # EI maximizes expected gain, PoI maximizes probability of *any* gain.
    # PoI is often more robust and focuses on finding *a* better point rather than *the best* point,
    # which can help in escaping plateaus where EI might be zero or tiny but PoI is still non-zero.
    
    # To address the "stagnation" and "local minima" issue, we can add a small exploration term.
    # Let's use a score that balances PoI and uncertainty.
    # Score = PoI + gamma * (sigma / max_sigma)
    # But we don't know max_sigma easily without scanning. 
    # Let's try: Score = PoI + 0.1 * sigma
    # This is effectively a UCB-like score but anchored on probability.
    
    # Wait, the cell finding explicitly says: "Adding a small exploration bonus to the 
    # probability-based acquisition function... Measured: 1.657 -> 1.672."
    # So the base should be PoI.
    
    exploration_bonus = 0.1 * sigma_safe
    scores = poi + exploration_bonus
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
