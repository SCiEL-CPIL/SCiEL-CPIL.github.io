"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard: 330s wall-clock limit, use 320s as safety margin
_T0 = time.time()
_DEADLINE = _T0 + 320.0

# Constants for the budget-aware strategy
BUDGET_LIMIT = 30.0
# Beta coefficient scaling factor. 
# LCB = mu + beta * sigma
# We want beta to be large when remaining budget is high, and small when low.
# beta = BETA_SCALE * sqrt(remaining_budget)
# At start (cost~0, rem~30), beta ~ BETA_SCALE * 5.47
# At end (cost~30, rem~0), beta ~ 0
# A typical UCB beta is around 1-5. 
# Let's set BETA_SCALE such that initial beta is around 2.0-3.0.
# 2.5 / 5.47 ~ 0.45
BETA_SCALE = 0.5

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return simple POI/cost
    # This is a fallback to ensure we always return valid scores quickly.
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Calculate remaining budget
    # cost is the cost of evaluating each candidate? 
    # The prompt says: "Evaluating x costs 1 + mean(x scaled to [0, 1]). Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
    # The input 'cost' has shape (2000,). This likely represents the cost of evaluating each of the 2000 candidates.
    # The loop maintains a 'total cost'. The acquisition function receives 'cost' which is the vector of costs for the candidates.
    # However, the acquisition function does NOT receive the current total cost directly.
    # Wait, the prompt says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
    # It does NOT pass the current total cost. 
    # How can we know the remaining budget?
    # The prompt says: "While total cost is below 30, the evaluator scores 2000 new uniform candidates..."
    # The acquisition function is called repeatedly. 
    # Is 'cost' the cumulative cost? No, it's shape (2000,). It's the cost of each candidate x.
    # We don't have access to the global state (total cost spent so far).
    # However, we can infer the "phase" of the search if we assume the function is called in a specific way or if we track state.
    # But the function is stateless in the interface.
    # Let's re-read carefully: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
    # There is no argument for current iteration or total cost.
    # This implies we cannot dynamically scale based on *actual* remaining budget unless we can estimate it or use a different strategy.
    
    # Alternative Interpretation:
    # Maybe we should use a strategy that doesn't require knowing the total cost, or we assume a standard progression?
    # Or perhaps the "cost" vector itself gives a clue? No, it's just the cost of the points.
    
    # Let's look at the inspiration: "budget-aware scaling ... aggressive exploration early ... precise exploitation later".
    # Without access to the total cost, we cannot implement true budget-aware scaling based on remaining budget.
    # However, we can use a proxy. 
    # In many BO loops, the number of iterations is roughly proportional to cost.
    # If we assume the loop runs until cost > 30, and average cost is ~1.5, we have ~20 iterations.
    # We can't count iterations in a stateless function.
    
    # Is there a way to encode the "phase" into the acquisition function without external state?
    # No.
    
    # Let's reconsider the "different strategy".
    # If we can't do budget-aware scaling due to missing state, what is a materially different strategy?
    # The current program uses POI/cost.
    # A different family is Expected Improvement (EI) or UCB.
    # EI is generally superior to POI for minimization.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) for minimization?
    # Let's check the sign.
    # We want to minimize.
    # Improvement I(x) = max(0, f_best - f(x)).
    # E[I(x)] = (f_best - mu) * P(f(x) < f_best) + sigma * phi(z)
    # where z = (f_best - mu) / sigma.
    # P(f(x) < f_best) = Phi(z).
    # phi(z) is the PDF of the standard normal.
    # So EI = (f_best - mu) * Phi(z) + sigma * phi(z).
    
    # Let's implement Cost-Aware EI.
    # Score = EI / cost.
    
    # This is a different design family (EI vs POI).
    # EI captures the magnitude of the expected improvement, not just the probability.
    # This is a standard and often stronger baseline than POI.
    
    # Let's verify the formula for minimization EI.
    # f(x) ~ N(mu, sigma^2).
    # I = max(0, f_best - f(x)).
    # E[I] = integral_{-inf}^{f_best} (f_best - y) * pdf(y) dy
    # Let y = mu + sigma * z. dy = sigma dz.
    # Limits: y=-inf -> z=-inf; y=f_best -> z = (f_best - mu)/sigma.
    # E[I] = integral_{-inf}^{z} (f_best - mu - sigma * t) * phi(t) / sigma * sigma dt
    # = integral_{-inf}^{z} (f_best - mu - sigma * t) * phi(t) dt
    # = (f_best - mu) * Phi(z) - sigma * integral_{-inf}^{z} t * phi(t) dt
    # integral_{-inf}^{z} t * phi(t) dt = -phi(z).
    # So E[I] = (f_best - mu) * Phi(z) + sigma * phi(z).
    
    # This looks correct.
    
    # Strategy: Cost-Aware Expected Improvement.
    # Score = EI / cost.
    
    # Why is this different from POI?
    # POI only considers the probability of improvement.
    # EI considers the expected amount of improvement.
    # Points with high uncertainty and low mean might have lower POI but higher EI if the potential drop is large.
    # This encourages exploring regions where a large drop is possible, not just likely.
    
    # Let's implement this.
    
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    # Note: if f_best - mu is negative, EI should be 0?
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) is always positive?
    # If mu > f_best, z is negative.
    # Let's check the limit.
    # The integral is always positive.
    # So we can just use the formula.
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
