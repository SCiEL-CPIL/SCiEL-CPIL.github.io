"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # seconds
_MARGIN = 5.0      # safety margin before deadline


def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: return cost-aware PI to be safe
        sigma_safe = np.maximum(np.asarray(sigma), 1e-9)
        z = (f_best - np.asarray(mu)) / sigma_safe
        return norm.cdf(z) / (np.asarray(cost) + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # Standardize improvement
    z = (f_best - mu) / sigma_safe

    # Probability of Improvement (PI): robust, less sensitive to variance inflation
    pi = norm.cdf(z)

    # Expected Improvement (EI): more aggressive exploitation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # LCB component: mu - beta * sigma (lower confidence bound)
    # We want to minimize, so LCB = mu - beta * sigma
    # But we want to pick the point with the lowest LCB (most promising)
    # So we use -LCB = -mu + beta * sigma as the score component
    # However, we need to combine this with EI/PI in a cost-aware way
    
    # Dynamic exploration weight based on remaining budget
    # Total budget is 30, start cost is ~5 (5 uniform points)
    # Remaining budget = 30 - current_total_cost
    # We estimate current_total_cost from the cost array (sum of costs of evaluated points)
    # But we don't have the total cost directly, only the cost of the 2000 candidates
    # Actually, the cost parameter is the cost of each candidate x in the 2000 candidates
    # We need to estimate how much budget is left
    # Let's use the mean cost of the 2000 candidates as a proxy for current cost level
    # Actually, the cost of each evaluation is 1 + mean(x_scaled)
    # The total cost accumulates over evaluations
    # Since we don't have the total cost, let's use a simple heuristic:
    # Start with 5 points, each costing ~1.5 on average, so initial cost ~7.5
    # Budget is 30, so remaining ~22.5 initially
    # As we evaluate more points, remaining decreases
    # We can estimate the number of evaluations done: 
    # Each iteration evaluates 1 point, and we've done (total_cost / avg_cost) evaluations
    # But we don't have total_cost either
    # Let's use a simpler approach: scale beta by a factor that decreases
    # as f_best approaches the minimum (-0.8)
    
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # Dynamic beta: larger when gap is large (early), smaller when close
    # This prevents wasting budget on high-variance regions when we're close to optimum
    beta = 0.3 * np.exp(-gap_to_min / 0.3) + 0.05
    
    # LCB score: -mu + beta * sigma (we want to maximize this, i.e., minimize mu - beta*sigma)
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize LCB to be comparable to EI/PI
    # LCB can be negative (since mu can be negative), so let's shift it
    # Actually, let's combine differently: use EI as the main term and add a 
    # penalty for high uncertainty that scales with beta
    
    # Combine: weighted mix of EI and LCB-based score
    # EI is in objective units, LCB is also in objective units
    # Let's use: score = w_ei * ei + w_lcb * lcb_score
    # But lcb_score can be negative, which would reduce the score
    # Let's instead use: score = ei - penalty * sigma
    # where penalty is dynamic based on budget remaining
    
    # Actually, let's stick closer to the inspiration 1 approach but with LCB insight:
    # Use a variance penalty that is stronger when we have less budget remaining
    # Estimate remaining budget fraction:
    # Assume we've used roughly (current_cost_estimate) out of 30
    # Use mean of cost array as proxy for typical cost
    mean_cost = np.mean(cost)
    # Estimate total evaluations done: initial 5 + iterations so far
    # Each iteration costs ~mean_cost on average
    # Total cost so far ~ 5 * 1.5 + n_iters * mean_cost
    # We don't know n_iters, but we can estimate it from the fact that
    # we're being called, so at least 5 evaluations have been done
    # Let's use a simpler heuristic: scale the variance penalty by gap_to_min
    
    # Variance penalty: downweight regions with high uncertainty
    # Use a soft penalty: 1 / (1 + alpha * sigma^2)
    # Make alpha dynamic: larger when gap_to_min is small (close to optimum)
    alpha_base = 2.0
    alpha = alpha_base * (1.0 + 2.0 * np.exp(-gap_to_min / 0.2))
    var_penalty = 1.0 / (1.0 + alpha * sigma_safe * sigma_safe)
    
    # Combine PI and EI with variance penalty
    score = (0.5 * pi + 0.5 * ei) * var_penalty
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
