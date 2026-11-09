"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s hard limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with
    Pareto-frontier filtering and Hypervolume contribution scoring.
    
    Args:
        mu (np.ndarray): Mean of GP posterior, shape (N,).
        sigma (np.ndarray): Std dev of GP posterior, shape (N,).
        f_best (float): Best observed function value.
        cost (np.ndarray): Cost of evaluating each candidate, shape (N,).
        
    Returns:
        np.ndarray: Acquisition scores, shape (N,). Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.random.uniform(0, 1, size=len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Compute UCB-like optimistic improvement score
    # This encourages exploration of high-uncertainty regions even if
    # their mean prediction is worse than f_best
    # Adaptive beta: larger when uncertainty is high relative to improvement
    sigma_med = np.median(sigma)
    improvement_scale = f_best - np.min(mu)
    if improvement_scale < 1e-9:
        improvement_scale = 1e-9
    
    # Beta that balances exploration vs exploitation
    # Larger beta when sigma is large relative to potential improvement
    beta = 2.0 * np.sqrt(np.log(1 + sigma_med / (improvement_scale + 1e-9)))
    beta = np.clip(beta, 0.5, 5.0)
    
    # UCB-based improvement (optimistic for minimization)
    ucb_imp = np.maximum(f_best - mu + beta * sigma, 0)
    
    # Also compute EI for blending
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend UCB and EI: when uncertainty is high relative to improvement,
    # trust UCB more
    w_ucb = sigma_med / (sigma_med + improvement_scale + 1e-9)
    w_ucb = np.clip(w_ucb, 0.3, 0.7)
    
    # Combined improvement score
    imp_score = (1 - w_ucb) * ei + w_ucb * ucb_imp
    
    # 2. Cost-aware scaling: divide by cost to get improvement per unit cost
    # Use a slight power to amplify cost sensitivity when exploring more
    gamma = 1.0 + 0.2 * w_ucb
    cost_aware_score = imp_score / (cost ** gamma)
    
    # Ensure finite values
    cost_aware_score = np.where(np.isfinite(cost_aware_score), cost_aware_score, 0.0)
    
    # 3. Sort candidates by cost (ascending) for Pareto frontier computation
    sort_indices = np.argsort(cost)
    sorted_cost = cost[sort_indices]
    sorted_score = cost_aware_score[sort_indices]
    
    # 4. Identify Pareto Frontier
    # A point is on the Pareto frontier if its score is strictly greater than
    # the max score of all points with lower or equal cost.
    max_score_so_far = np.maximum.accumulate(sorted_score)
    is_pareto = (sorted_score == max_score_so_far)
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.where(np.isfinite(cost_aware_score), cost_aware_score, 0.0)
    
    # 5. Calculate Hypervolume Contribution for Pareto Points
    C_max = 2.0  # Maximum possible cost (1 + max mean(x) where x in [0,1])
    
    # Extract Pareto points
    pareto_indices_in_sorted = np.where(is_pareto)[0]
    
    if len(pareto_indices_in_sorted) == 0:
        return np.where(np.isfinite(cost_aware_score), cost_aware_score, 0.0)
    
    pareto_scores_arr = sorted_score[pareto_indices_in_sorted]
    pareto_costs = sorted_cost[pareto_indices_in_sorted]
    
    # Calculate "heights" (difference in score from previous Pareto point)
    prev_pareto_scores = np.zeros_like(pareto_scores_arr)
    if len(pareto_scores_arr) > 1:
        prev_pareto_scores[1:] = pareto_scores_arr[:-1]
    
    heights = pareto_scores_arr - prev_pareto_scores
    
    # Calculate "widths" (cost range covered by this point's contribution)
    widths = C_max - pareto_costs
    
    # HV Contribution Score
    hv_scores = heights * widths
    
    # 6. Map scores back to original indices
    final_scores = np.where(np.isfinite(cost_aware_score), cost_aware_score, 0.0)
    original_pareto_indices = sort_indices[pareto_indices_in_sorted]
    final_scores[original_pareto_indices] = hv_scores
    
    # Final safety: ensure all values are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
