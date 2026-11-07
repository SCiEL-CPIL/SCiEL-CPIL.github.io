"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def _pareto_frontier(mu, sigma, cost, f_best):
    """
    Compute a Pareto-frontier score based on non-dominated sort of
    (predicted improvement, uncertainty, cost).
    
    Returns a score array where points on the Pareto frontier get high scores,
    and points dominated by others get lower scores.
    """
    n = len(mu)
    
    # Objective 1: Maximize predicted improvement (f_best - mu)
    imp = f_best - mu
    
    # Objective 2: Maximize uncertainty (sigma) - for exploration
    unc = sigma.copy()
    
    # Objective 3: Minimize cost
    # For Pareto, we want to maximize -cost, so negate it
    neg_cost = -cost.copy()
    
    # Non-dominated sort: assign rank 0 to Pareto-optimal, rank 1 to next layer, etc.
    # For efficiency with 2000 points, use a simple approach
    ranks = np.zeros(n, dtype=np.int32)
    remaining = np.ones(n, dtype=bool)
    
    for rank_level in range(n):
        if not remaining.any():
            break
        
        # Find non-dominated points among remaining
        # A point i dominates point j if:
        # imp[i] >= imp[j] AND unc[i] >= unc[j] AND neg_cost[i] >= neg_cost[j]
        # AND at least one is strict
        
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) == 0:
            break
        
        # For each remaining point, check if it's dominated by any other remaining point
        # This is O(n^2) worst case but n=2000 so ~4M ops, should be fast
        
        dominated = np.zeros(len(rem_idx), dtype=bool)
        
        for i in range(len(rem_idx)):
            if dominated[i]:
                continue
            idx_i = rem_idx[i]
            imp_i = imp[idx_i]
            unc_i = unc[idx_i]
            nc_i = neg_cost[idx_i]
            
            for j in range(len(rem_idx)):
                if i == j or dominated[j]:
                    continue
                idx_j = rem_idx[j]
                imp_j = imp[idx_j]
                unc_j = unc[idx_j]
                nc_j = neg_cost[idx_j]
                
                # Check if i dominates j
                if imp_i >= imp_j and unc_i >= unc_j and nc_i >= nc_j:
                    # At least one must be strict
                    if imp_i > imp_j or unc_i > unc_j or nc_i > nc_j:
                        dominated[j] = True
        
        # Assign this rank level to non-dominated remaining points
        for i in range(len(rem_idx)):
            if not dominated[i]:
                ranks[rem_idx[i]] = rank_level
                remaining[rem_idx[i]] = False
        
        if rank_level > 50:  # Safety: don't go too deep
            break
    
    # Points not assigned get the max rank
    max_rank = np.max(ranks)
    unassigned = ranks == 0
    # Actually, let's be more careful. Initialize ranks to -1
    # Let me redo this more carefully
    
    # Reset and do it properly
    ranks = np.full(n, -1, dtype=np.int32)
    remaining = np.ones(n, dtype=bool)
    
    for rank_level in range(min(20, n)):  # Limit to 20 levels for speed
        if not remaining.any():
            break
        
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) == 0:
            break
        
        dominated = np.zeros(len(rem_idx), dtype=bool)
        
        # Vectorized dominance check for speed
        # For each pair, check dominance
        # This is still O(n^2) but with vectorization over j for each i
        
        for i in range(len(rem_idx)):
            if dominated[i]:
                continue
            idx_i = rem_idx[i]
            imp_i = imp[idx_i]
            unc_i = unc[idx_i]
            nc_i = neg_cost[idx_i]
            
            # Find all j that i dominates
            imp_j = imp[rem_idx]
            unc_j = unc[rem_idx]
            nc_j = neg_cost[rem_idx]
            
            dom_mask = (imp_i >= imp_j) & (unc_i >= unc_j) & (nc_i >= nc_j)
            strict_mask = (imp_i > imp_j) | (unc_i > unc_j) | (nc_i > nc_j)
            
            # i dominates j if dom_mask and strict_mask, and i != j
            dom_mask[i] = False  # Don't dominate self
            dominated_j = dom_mask & strict_mask
            
            # Mark dominated points
            dominated[dominated_j] = True
        
        # Assign rank to non-dominated
        for i in range(len(rem_idx)):
            if not dominated[i]:
                ranks[rem_idx[i]] = rank_level
                remaining[rem_idx[i]] = False
    
    # Unassigned points get a high rank
    unassigned = ranks < 0
    if unassigned.any():
        ranks[unassigned] = 20
    
    # Convert ranks to scores: lower rank = better
    # Normalize: score = 1 / (1 + rank)
    pf_scores = 1.0 / (1.0 + ranks.astype(np.float64))
    
    return pf_scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Combines Pareto-frontier selection with cost-aware UCB/EI blending.
    
    Parameters
    ----------
    mu : array (2000,) GP mean
    sigma : array (2000,) GP std
    f_best : float best observed value
    cost : array (2000,) evaluation cost per candidate
    
    Returns
    -------
    scores : array (2000,) higher = evaluate next
    """
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s wall-clock limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0)
    
    n = len(mu)
    
    # --- Phase 1: Compute base acquisition scores ---
    
    # GP-UCB for minimization
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-9:
        mu_range = 1.0
    
    sigma_rel = np.median(sigma) / mu_range
    
    beta_max = 5.0
    beta_min = 0.5
    beta_t = beta_min + (beta_max - beta_min) * np.clip(sigma_rel / 1.0, 0, 1)
    
    cost_mean = np.mean(cost)
    cost_factor = 1.0 / (1.0 + 0.1 * (cost_mean - 1.0))
    beta_t *= cost_factor
    
    beta_cost = 0.7  # Tighter cost exponent from knowledge
    ucb_term = -mu + beta_t * sigma
    cost_normalizer = cost ** beta_cost
    ucb_scores = ucb_term / cost_normalizer
    
    # EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # --- Phase 2: Pareto frontier computation ---
    
    if time.time() - t0 > DEADLINE - 100:
        # Skip Pareto if we're running out of time
        pf_scores = np.ones(n)
    else:
        pf_scores = _pareto_frontier(mu, sigma, cost, f_best)
    
    # --- Phase 3: Combine ---
    
    # Blend UCB and EI based on stage
    ei_weight = np.clip(1.0 - sigma_rel, 0.3, 0.7)
    ucb_weight = 1.0 - ei_weight
    
    # Normalize UCB scores
    ucb_abs_max = np.max(np.abs(ucb_term))
    if ucb_abs_max > 1e-9:
        ucb_norm = ucb_term / ucb_abs_max
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    blended = ei_weight * ei_norm + ucb_weight * ucb_norm
    blended_cost = blended / cost_normalizer
    
    # Pareto boost: points on the frontier get a bonus
    # The Pareto score is in [~0.05, 1.0]
    # Boost factor: 20% additional weight for frontier points
    pareto_boost = 0.2
    final_scores = blended_cost * (1.0 + pareto_boost * pf_scores)
    
    # Ensure finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # If Pareto computation was skipped, fall back to the blended score
    if np.all(pf_scores == 1.0):
        final_scores = blended_cost
    
    # Final finite check
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
