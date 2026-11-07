"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track elapsed time for adaptive strategy
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety guard: if we are dangerously close to the 330s limit, 
    # return a simple, fast score to ensure we don't timeout.
    # The evaluator kills after 330s. We want to be safe.
    current_time = time.time()
    elapsed = current_time - _start_time
    
    # Basic finite check
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate standard EI
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Strategy: Adaptive Exploration (xi) and Cost Penalty
    # 
    # The "Three Hump Camel" has a global minimum at (0,0) [value 0] 
    # and a local minimum near (-0.3, 0.3) [value ~0.0007].
    # In the scaled space [0,1]^2:
    # (0,0) -> (0.5, 0.5). Cost = 1 + mean([0.5, 0.5]) = 1.5.
    # (-0.3, 0.3) -> (0.47, 0.53). Cost = 1 + mean([0.47, 0.53]) = 1.5.
    # The costs are nearly identical. The main risk is getting stuck in the local minimum.
    #
    # Early on, we want to explore to find the basin of the global minimum.
    # Later, we want to exploit.
    #
    # The previous program used Thompson Sampling with a random seed (implicit in np.random).
    # This introduces variance. A deterministic EI with an adaptive xi might be more stable.
    #
    # We use a "soft" cost penalty. Dividing by cost directly is too harsh if costs vary slightly.
    # Instead, we subtract a small multiple of cost, or divide by a power.
    # Given the finding "Reducing the cost penalty bias helps", we should be very gentle with cost.
    #
    # Let's try: Score = EI - lambda * cost
    # Or: Score = EI / (1 + lambda * (cost - min_cost))
    #
    # Actually, a simpler and very effective strategy for small budgets is 
    # "Expected Improvement with a penalty for cost" where the penalty is tuned.
    #
    # Let's look at the budget.
    # Start: 5 points. Cost ~ 7.5.
    # Budget: 30.
    # Remaining budget: ~22.5.
    # Average cost per eval: ~1.5.
    # We can afford ~15 more evaluations.
    #
    # If we use EI / cost, we prioritize cheap points.
    # If the global min is slightly more expensive (due to noise in the "uniform point in inner 80%" shift),
    # we might miss it.
    #
    # Let's use a hybrid:
    # 1. Compute EI.
    # 2. Apply a mild cost penalty: score = EI - 0.1 * cost
    #    Wait, EI can be large if f_best is far. Cost is ~1-2.
    #    If EI is 0.01 and cost is 1.5, penalty 0.15 kills it.
    #    We need to scale the penalty.
    #
    # Alternative: Use UCB.
    # UCB = -mu + beta * sigma
    # We want to maximize UCB (minimize -UCB).
    # But UCB doesn't use f_best directly, which can be bad if we don't have a good f_best yet.
    # EI uses f_best, which is good for exploitation.
    #
    # Let's stick to EI but modify the "improvement" definition or the cost handling.
    #
    # Idea: "Cost-Adjusted EI"
    # The probability of improvement is P(f(x) < f_best).
    # The expected cost is cost(x).
    # We want to maximize E[Improvement] / E[Cost]?
    # Or E[Improvement] - lambda * Cost?
    #
    # Let's try a dynamic lambda.
    # Early on (low call count), lambda is small (explore/ignore cost).
    # Late on, lambda is larger (save budget).
    #
    # But we don't know the exact call count from inside acquisition easily without global state.
    # We have _call_count.
    
    _call_count += 1
    
    # Determine stage
    # Total budget 30. Initial cost ~7.5.
    # If we assume average cost 1.5, we have ~15 calls left.
    # Total calls ~20.
    
    if _call_count <= 5:
        # Early phase: Explore. Minimize cost penalty.
        # Use EI / (1 + 0.1 * cost)
        scores = ei / (1.0 + 0.1 * cost)
    elif _call_count <= 15:
        # Mid phase: Balance.
        scores = ei / (1.0 + 0.2 * cost)
    else:
        # Late phase: Exploit. Stronger cost penalty to save budget?
        # Or just pure EI if we are close to the minimum?
        # If we are close, sigma is low, EI is small. Cost penalty might dominate.
        # Let's keep it mild.
        scores = ei / (1.0 + 0.3 * cost)
        
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we are very close to timeout, return a simple score to be safe
    # (Though the vectorized ops above are fast, this is a just-in-case)
    if elapsed > 325.0:
         # Fallback to simple -mu (exploitation)
         return -mu / (1.0 + 0.1 * cost)

    return scores
# EVOLVE-BLOCK-END
