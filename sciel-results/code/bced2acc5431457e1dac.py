import numpy as np
from scipy.stats import norm
import time

_t0 = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE:
        # Emergency fallback: pure exploitation with cost awareness
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return np.maximum(score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n == 0:
        return np.array([])

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)

    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Dynamic exploration-exploitation balance
    # Total budget is 30.0. Cost so far is tracked via the 'cost' parameter.
    # The cost of the NEXT evaluation is in the 'cost' array (each candidate's cost).
    # We estimate remaining budget as 30.0 - current_total_cost.
    # But we don't have total cost directly. We have cost per candidate.
    # The loop runs while total_cost < 30. Each eval costs 1 + mean(x_scaled).
    # Since we're in the acquisition function, we know the cost of each candidate.
    # Let's estimate how many more evaluations we can do.
    # Average cost of a uniform point in [0,1]^4 is 1 + 0.5 = 1.5.
    # So roughly (30 - total_spent) / 1.5 more evaluations.
    # We don't know total_spent exactly, but we can infer from the number of
    # evaluations already done. The loop starts with 5 points, then adds 1 per iteration.
    # Actually, the cost array gives us the cost of each NEW candidate.
    # Let's use a simpler heuristic: the fraction of budget remaining.
    # If we've spent a lot, we should exploit more.
    
    # A robust way: use the current f_best and the scale of the problem.
    # When f_best is close to 0, we're near the optimum, so exploit.
    # When f_best is large, explore.
    
    # Also, use the average cost of candidates to estimate budget usage.
    # If average cost is high, we're burning budget fast.
    
    # Let's use a time-based and f_best-based adaptive beta.
    # Early on (high f_best), explore more. Late (low f_best), exploit more.
    
    f_best_scale = max(abs(f_best), 1e-6)
    
    # Beta for UCB: higher when f_best is large (early stage)
    # Beta ranges from ~2.0 (early) to ~0.5 (late)
    beta = 0.5 + 1.5 * (1.0 / (1.0 + f_best_scale))
    
    # Cost-awareness: penalize expensive points more when budget is tight
    # Estimate remaining budget: assume average cost ~1.5, and we've done
    # roughly (current_iteration) evaluations. But we don't have iteration count.
    # Instead, use the fact that if f_best is small, we're late.
    # Cost weight: inverse of cost, but damped.
    # cost_weight = 1 / (1 + cost_safe) would be too strong.
    # Let's use cost_weight = 1 / cost_safe**gamma where gamma adapts.
    
    # When f_best is large (early), cost matters less (gamma small).
    # When f_best is small (late), cost matters more (gamma large).
    # Increase base gamma to be more cost-aware early on to save budget.
    gamma = 0.6 + 0.4 * (1.0 / (1.0 + f_best_scale))
    
    cost_weight = cost_safe ** (-gamma)
    
    # UCB term for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Combine: weighted EI + weighted UCB
    # The EI term is in objective units, UCB is also in objective units.
    # Let's blend them.
    
    # Primary: cost-weighted EI
    score_ei = ei * cost_weight
    
    # Secondary: cost-weighted UCB with adaptive weight
    # UCB weight: higher when exploring (early), lower when exploiting (late)
    # Slightly increased base weight to avoid premature convergence.
    ucb_weight = 0.2 * (1.0 / (1.0 + f_best_scale))
    score_ucb = ucb_weight * ucb * cost_weight
    
    score = score_ei + score_ucb
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
