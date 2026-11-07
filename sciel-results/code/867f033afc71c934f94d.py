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
    
    for rank_ in range(n):
        if not np.any(remaining):
            break
        # Get indices of remaining points
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) == 0:
            break
        if len(rem_idx) == 1:
            ranks[rem_idx] = rank_
            remaining[rem_idx] = False
            continue
        
        # Get the objective vectors for remaining points
        imp_rem = imp[rem_idx]
        unc_rem = unc[rem_idx]
        nc_rem = neg_cost[rem_idx]
        
        # A point is dominated if there exists another point that is >= in all objectives
        # and > in at least one
        # For efficiency, use vectorized approach
        # imp_rem[:, None] >= imp_rem[None, :] etc.
        
        # Create dominance matrix: D[i,j] = True if i dominates j
        # i dominates j if imp_i >= imp_j and unc_i >= unc_j and nc_i >= nc_j and at least one strict
        imp_dom = imp_rem[:, None] >= imp_rem[None, :]
        unc_dom = unc_rem[:, None] >= unc_rem[None, :]
        nc_dom = nc_rem[:, None] >= nc_rem[None, :]
        
        all_dom = imp_dom & unc_dom & nc_dom
        strict = (imp_rem[:, None] > imp_rem[None, :]) | \
                 (unc_rem[:, None] > unc_rem[None, :]) | \
                 (nc_rem[:, None] > nc_rem[None, :])
        
        dominated = all_dom & strict
        
        # A point is non-dominated if no other point dominates it
        # i.e., no j such that j dominates i
        # dominated[i, j] = True means i dominates j
        # So j is dominated if any i dominates j, i.e., dominated[:, j].any()
        is_dominated = dominated.any(axis=0)
        
        # Non-dominated points in this remaining set
        non_dom_mask = ~is_dominated
        non_dom_local = rem_idx[non_dom_mask]
        
        if len(non_dom_local) > 0:
            ranks[non_dom_local] = rank_
            remaining[non_dom_local] = False
        else:
            # This shouldn't happen, but break to avoid infinite loop
            break
    
    # Convert ranks to scores: lower rank = better
    # Normalize ranks to [0, 1] with 0 being best
    max_rank = np.max(ranks)
    if max_rank > 0:
        rank_score = 1.0 - ranks / (max_rank + 1)
    else:
        rank_score = np.ones(n)
    
    return rank_score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a Pareto-frontier based approach combined with cost-aware EI.
    
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
    
    # Compute cost-aware Expected Improvement
    # Standard EI for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to get expected improvement per unit cost
    ei_cost = ei / cost
    
    # Compute Pareto frontier scores
    pareto_scores = _pareto_frontier(mu, sigma, cost, f_best)
    
    # Normalize EI scores to [0, 1]
    ei_max = np.max(ei_cost)
    if ei_max > 1e-9:
        ei_norm = ei_cost / ei_max
    else:
        ei_norm = np.zeros_like(ei_cost)
    
    # Combine: use a weighted blend of Pareto frontier position and cost-aware EI
    # Points on the Pareto frontier that also have high cost-aware EI get the highest scores
    # The Pareto score already captures the trade-off, so we can multiply or blend
    
    # Multiply: a point needs to be both on the frontier AND have high EI to score well
    scores = pareto_scores * ei_norm
    
    # Also add a small baseline to ensure all points have some score
    # This helps with exploration in early stages
    scores += 0.1 * ei_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
