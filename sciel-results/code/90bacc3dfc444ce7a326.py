import numpy as np
import time

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s limit, leave 10s margin for safety
    if time.time() - _start_time > 320:
        # If we are running out of time, just return a simple deterministic score
        # to ensure we return valid output quickly.
        # This fallback is unlikely to be hit if the logic is fast, but it's a safety net.
        return -mu / cost

    # 1. Calculate the adaptive exploration parameter beta.
    # We want to explore more when we are uncertain (high sigma) or when f_best is far from 0 (early phase).
    # A common heuristic is to scale beta with the maximum uncertainty or use a fixed schedule.
    # Given the small budget (cost < 30), we are in a very early phase of BO.
    # The Ackley function has a global minimum at 0.
    
    # Let's use an adaptive beta that starts high (exploration) and decreases as we get closer to 0.
    # max_sigma is a proxy for how much we don't know.
    max_sigma = np.max(sigma)
    
    # If sigma is very small, we should exploit. If large, explore.
    # We can also look at f_best. If f_best is high (far from min 0), we need to explore more.
    # Let's define a "confidence" level.
    
    # A robust approach: 
    # LCB = mu - beta * sigma
    # We want to maximize -LCB = -mu + beta * sigma.
    # Score = (-mu + beta * sigma) / cost
    
    # How to choose beta?
    # In the inspiration 2 (score 3.45), it mentioned "cost-awareness mechanism that uses the GP uncertainty".
    # Let's try a beta that scales with the current best value's distance from 0, normalized by the scale of the function.
    # Ackley 2D range is roughly [0, 20].
    # If f_best is 10, we are far. If f_best is 0.1, we are close.
    
    # Let's use a simple adaptive strategy:
    # beta = max(0.5, min(5.0, 2.0 * max_sigma / (f_best + 1e-6)))
    # This increases beta if sigma is large relative to the remaining potential improvement.
    
    # However, a simpler and very effective strategy for small budgets is a fixed, moderately aggressive beta.
    # The "Inspiration 1" (score 2.36) used LCB with cost-awareness.
    # The "Inspiration 2" (score 3.45) refined this.
    
    # Let's try a "Dynamic LCB" where beta is proportional to the standard deviation of the observed values 
    # or simply a fixed value that has worked well in similar constrained BO tasks.
    # Given the cost structure (1 + mean(x_scaled)), points near the center (0,0) in scaled space 
    # (which is [2.5, 2.5] in original coords? No, [-5,5] scaled to [0,1] means 0 is 0.5) 
    # have lower cost.
    # Wait, x_scaled = (x - (-5)) / 10. 
    # If x is in [-5, 5], x_scaled is in [0, 1].
    # Cost = 1 + mean(x_scaled).
    # Min cost is 1 (when x is at -5, -5). Max cost is 2 (when x is at 5, 5).
    # The global minimum of Ackley is at (0,0).
    # (0,0) scaled is (0.5, 0.5). Cost = 1 + 0.5 = 1.5.
    # So the global minimum is NOT the cheapest point. The cheapest points are in the corner (-5, -5).
    # This is a tricky cost structure. We must balance finding the low mu (Ackley value) with low cost.
    
    # Since the minimum is at cost 1.5, and max cost is 2, the cost variation is only 1.5x to 2x.
    # The value variation is much larger.
    
    # Let's use a fixed beta of 2.0. This is a common choice for LCB in 2D.
    # Score = (beta * sigma - mu) / cost
    
    # To make it "adaptive" and potentially better:
    # If f_best is very good (close to 0), we should reduce beta to exploit.
    # If f_best is poor, increase beta to explore.
    
    # Let's try: beta = 1.0 + 2.0 * (f_best / 5.0)
    # If f_best = 0, beta = 1.0 (exploit).
    # If f_best = 5, beta = 3.0 (explore).
    # If f_best = 20 (max), beta = 9.0 (very explore).
    
    # Let's cap beta to avoid extreme exploration.
    beta = 1.0 + 2.0 * (f_best / 5.0)
    beta = np.clip(beta, 1.0, 5.0)
    
    # Calculate LCB score
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
