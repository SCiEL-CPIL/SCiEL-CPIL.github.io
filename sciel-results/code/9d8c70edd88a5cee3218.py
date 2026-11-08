import numpy as np
from scipy.stats import norm
import time

# Deadline guard constants
START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Effective Upper Confidence Bound (CE-UCB).
    Returns scores for 2000 candidates. Higher score = evaluate next.
    """
    # Check time budget
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to a simple heuristic if we are running out of time
        # Return inverse cost (prefer cheap points)
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    safe_sigma = np.where(sigma < eps, eps, sigma)
    
    # Cost normalization: Lower cost is better.
    # We want to divide by cost to penalize expensive evaluations.
    # The cost is defined as 1 + mean(x_scaled). Range [1, 2].
    safe_cost = np.where(cost < eps, eps, cost)
    
    # Dynamic Exploration Parameter (kappa)
    # The 'cost' argument here is the cost of the *candidates*, not the total budget.
    # However, we can infer progress from f_best or simply use a fixed reasonable kappa.
    # In standard UCB, kappa is often sqrt(2 * log(t)). 
    # Since we don't have 't' (step count) directly, but we have f_best,
    # we can use a heuristic: start with high exploration, decay it.
    # A common static value for 4D problems is around 1.0 to 2.0.
    # Let's use a slightly adaptive approach based on the current best.
    # If f_best is still very high, explore more. If f_best is close to 0, exploit.
    
    # Powell 4D minimum is 0.
    # If f_best is large, we are far from optimum. High kappa.
    # If f_best is small, we are close. Low kappa.
    # Scale: f_best usually starts around 10-20 and goes to 0.
    # Let's map f_best to kappa.
    # kappa = 2.0 * exp(-0.1 * f_best) + 0.5
    # If f_best=20, kappa ~ 2.0 * 0.135 + 0.5 = 0.77
    # If f_best=0, kappa ~ 2.0 * 1.0 + 0.5 = 2.5 (Wait, this is backwards)
    
    # Correction: We want HIGH exploration when f_best is BAD (high value).
    # So if f_best is high, kappa should be high.
    # If f_best is low (good), kappa should be low.
    # kappa = 1.0 + 1.0 * (f_best / (f_best + 1.0))
    # If f_best = 10, kappa = 1 + 10/11 = 1.9
    # If f_best = 0.1, kappa = 1 + 0.1/1.1 = 1.09
    # If f_best = 0, kappa = 1.0
    
    kappa = 1.0 + 1.0 * (f_best / (f_best + 1.0))
    
    # Calculate UCB score: mu + kappa * sigma
    # Since we are minimizing, we want to find points with LOW potential.
    # UCB usually maximizes mu + kappa*sigma for maximization.
    # For minimization, we can use Lower Confidence Bound (LCB): mu - kappa * sigma.
    # Or we can maximize -(mu - kappa * sigma) = -mu + kappa * sigma.
    # Let's stick to the minimization convention: Score = mu - kappa * sigma.
    # Higher score in our return value means "better to pick".
    # So we want to pick the candidate with the LOWEST LCB.
    # Therefore, our "score" to return should be negative of LCB, or we invert the logic.
    # The task says: "Return 2000 finite scores (higher = evaluate next)".
    # So if I calculate LCB = mu - kappa*sigma, I want the MIN LCB.
    # So I should return -LCB = -mu + kappa*sigma.
    
    lcb = mu - kappa * safe_sigma
    scores = -lcb
    
    # Cost-Aware Scoring
    # Divide by cost to penalize expensive points.
    # scores = -lcb / cost
    # This effectively asks: "How much improvement (in terms of lower bound) do I get per unit cost?"
    
    scores = scores / safe_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
