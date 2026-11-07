"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Global Budget-Aware UCB."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_EPS = 1e-9
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit
_BUDGET_LIMIT = 30.0
_INIT_COST = 5.0  # Approximate initial cost of 5 uniform points (avg cost ~1.5 each)

def acquisition(mu, sigma, f_best, cost):
    """
    Global Budget-Aware UCB acquisition function.
    
    Strategy:
    1. Instead of dividing by local cost (which can create noise when costs are similar 
       but not identical, or over-penalize slightly higher cost regions), we use a 
       UCB score: -mu + beta * sigma.
    2. We subtract a small static penalty proportional to local cost to handle 
       cost-awareness gently: -gamma * cost.
    3. The key structural change is that beta (the exploration coefficient) is 
       dynamic. It depends on the remaining budget.
       - Early in the run (high remaining budget), beta is high to encourage exploration.
       - Late in the run (low remaining budget), beta is low to encourage exploitation.
       
    This mimics the idea that we can afford to be wrong (explore) when we have lots 
    of evaluations left, but must be sure (exploit) when we are close to the limit.
    
    The cost passed to the function is the cumulative cost array. 
    The maximum value in cost represents the current total cost consumed by the 
    evaluator loop (since the evaluator picks the best of 2000, the cost of the 
    next evaluation is determined by the loop state, but here we are given costs 
    for the 2000 candidates. The "current total cost" is effectively the cost 
    accumulated so far. In this specific interface, `cost` is the cost of 
    evaluating each of the 2000 candidates. The total cost consumed so far is 
    not directly passed as a scalar, but we can infer progress.
    
    Wait, the task says: "mu, sigma and cost have shape (2000,). ... Evaluating x 
    costs 1 + mean(x scaled to [0, 1]). Runs start with 5 uniform random points 
    (cost counted). While total cost is below 30, the evaluator scores 2000 new 
    uniform candidates... and evaluates the best one."
    
    The `cost` array contains the cost of evaluating each of the 2000 *candidate* 
    points. It does NOT contain the history of past evaluations. 
    However, we can estimate the current iteration or total cost consumed if we 
    had state, but we don't. 
    
    Can we infer the total cost consumed?
    No, `f_best` is the only state variable passed that changes. 
    But `f_best` doesn't tell us how much cost has been spent.
    
    Let's re-read carefully: "mu, sigma and cost have shape (2000,). Return 2000 
    finite scores... No coordinates are passed."
    
    Is it possible to track state? The function is called repeatedly. 
    We can use a global variable to track the number of calls or the last f_best.
    However, `f_best` might not change if the new point is worse.
    
    Actually, a simpler structural change that fits the "global budget penalty" 
    hint without needing external state is to use the *distribution* of costs or 
    the *mean* cost of the candidates as a proxy for the "typical" cost, but that 
    doesn't track global budget.
    
    Let's look at the inspiration: "subtracting a term proportional to the current 
    total cost consumed (or remaining budget ratio)".
    
    If I cannot access the total cost consumed directly, I can approximate the 
    progress. 
    The loop runs while total_cost < 30.
    Initial cost = ~5 (5 points * ~1.5).
    Each step adds ~1.5 cost on average.
    Total steps ~ (30 - 5) / 1.5 = ~16 steps.
    
    If I use a global counter for the number of calls, I can estimate the 
    remaining budget.
    Call 0: Initial.
    Call 1: After 1 eval.
    ...
    
    Let's implement a global counter. This is a stateful change, which is a 
    structural shift from a stateless function.
    
    Beta schedule:
    Let N_calls be the number of times acquisition has been called.
    Estimated total cost consumed = 5 + N_calls * 1.5 (approx).
    Remaining budget = 30 - estimated_total_cost.
    Ratio = remaining / 25 (initial remaining).
    Beta = beta_max * (Ratio)^alpha.
    
    This dynamically shifts from exploration to exploitation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Global state to track progress
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    
    # Increment call count
    acquisition._call_count += 1
    current_call = acquisition._call_count
    
    # Estimate total cost consumed
    # Start: 5 points. Average cost of uniform point in [-4,5]^4 scaled to [0,1] is 0.5.
    # Cost = 1 + 0.5 = 1.5.
    # Initial cost = 5 * 1.5 = 7.5? 
    # "Runs start with 5 uniform random points (cost counted)."
    # Let's assume initial cost is roughly 5 * 1.5 = 7.5.
    # Or maybe the 5 points are just the seed, and the loop starts after.
    # Let's assume total_cost_so_far = 7.5 + (current_call - 1) * 1.5
    
    est_initial_cost = 7.5
    avg_step_cost = 1.5
    est_total_cost = est_initial_cost + max(0, current_call - 1) * avg_step_cost
    
    # Clamp to reasonable bounds
    est_total_cost = np.clip(est_total_cost, 0, _BUDGET_LIMIT)
    
    remaining_budget = _BUDGET_LIMIT - est_total_cost
    initial_remaining = _BUDGET_LIMIT - est_initial_cost
    
    # Ratio of remaining budget to initial remaining budget
    if initial_remaining > 0:
        ratio = remaining_budget / initial_remaining
    else:
        ratio = 0.0
        
    ratio = np.clip(ratio, 0.0, 1.0)
    
    # Dynamic Beta
    # Start with high exploration (beta=3.0) and decay to low exploitation (beta=0.5)
    beta_max = 3.0
    beta_min = 0.5
    # Use a power law to decay faster in the middle
    beta = beta_min + (beta_max - beta_min) * (ratio ** 0.5)
    
    # Cost penalty gamma
    # We want to penalize higher costs, but not as aggressively as division.
    # The scale of (f_best - mu) is in objective units.
    # The scale of cost is ~1-2.
    # A small gamma, e.g., 0.1, might be appropriate.
    gamma = 0.1
    
    # UCB Score for minimization:
    # We want to minimize Y.
    # Uncertainty bonus: + beta * sigma
    # Improvement expectation: -mu
    # Cost penalty: - gamma * cost
    
    score = -mu + beta * sigma - gamma * cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END
