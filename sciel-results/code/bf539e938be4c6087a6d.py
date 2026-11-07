import numpy as np
import time

# Global start time to track elapsed time for deadline safety
_start_time = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimizing Hartmann 6D.
    
    Args:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, best observed value so far
    cost: (2000,) array of evaluation costs (1 + mean(x))
    
    Returns:
    (2000,) array of scores, higher is better
    """
    # Deadline guard: if we are close to the limit, return a safe fallback
    # This is unlikely to trigger in a single call, but protects against cumulative delays
    if time.time() - _start_time > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and add small epsilon for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # But acquisition functions usually return "higher is better".
    # Standard UCB for maximization is mu + beta * sigma.
    # For minimization, we can think of maximizing the potential for improvement.
    # A common cost-aware formulation is: Score = (f_best - mu) / cost + beta * sigma / cost
    # However, the directive suggests: UCB(x) = mu(x) + beta sigma(x) ... divide by cost.
    # Wait, for minimization, lower mu is better.
    # If we use standard UCB for maximization of a "reward" where reward = -f(x),
    # then mu_reward = -mu_f, sigma_reward = sigma_f.
    # UCB_reward = -mu_f + beta * sigma_f.
    # So we want to maximize (-mu + beta * sigma) / cost.
    
    # Let's verify the sign.
    # We want to find x that minimizes f(x).
    # If mu is low, it's a good candidate for exploitation.
    # If sigma is high, it's a good candidate for exploration.
    # So the "score" should be high when mu is low or sigma is high.
    # Score = (-mu + beta * sigma) / cost
    
    # Adaptive beta: 
    # Early on (low n), we want more exploration (higher beta).
    # Later (high n), we want more exploitation (lower beta).
    # We don't have n directly, but we can estimate it from the magnitude of f_best 
    # or just use a fixed beta if we don't have iteration count.
    # The prompt says "While total cost is below 30...". 
    # Initial cost is 5 points. Max cost 30.
    # We can infer the "stage" from f_best. 
    # Hartmann 6D min is ~ -3.32.
    # If f_best is very high (e.g., > -1.0), we are in early exploration.
    # If f_best is close to -3.32, we are in late exploitation.
    
    # Let's use a heuristic for beta based on f_best.
    # f_best ranges roughly from -1.0 to -3.32.
    # Map f_best to a beta range, e.g., [4.0, 1.0].
    # beta = 4.0 - 3.0 * (f_best - (-1.0)) / (-3.32 - (-1.0))
    # This is fragile if f_best is outside this range.
    
    # Alternatively, a fixed beta is safer. The directive suggests "fixed exploration constant (e.g., 2.0)".
    # Let's try a slightly higher beta initially, but since we can't track iterations reliably 
    # without state, we will stick to a fixed beta or a simple heuristic.
    # Given the short budget (cost < 30, starting at 5, so ~25 evaluations),
    # exploration is important early on.
    # Let's use beta = 2.5.
    
    beta = 2.5
    
    # Calculate UCB score for minimization
    # We want to maximize: (-mu + beta * sigma) / cost
    # Note: mu and sigma are in objective units.
    # cost is in [1, 2].
    
    numerator = -mu + beta * sigma_safe
    denominator = np.maximum(cost, 1e-8)
    
    scores = numerator / denominator
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
