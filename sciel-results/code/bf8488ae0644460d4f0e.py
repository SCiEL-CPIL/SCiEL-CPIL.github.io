"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Structural change: Model the search as a constrained stochastic shortest path
    problem by computing Expected Improvement per unit cost (EIPC) and applying
    a dynamic threshold that rejects candidates with low EIPC, thereby explicitly
    pruning the 2000-candidate pool to only those that offer the best value-for-money
    trade-off. This shifts from continuous scoring to a discrete, budget-feasible
    selection.
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = t_start + 280  # 300s limit, keep 20s safety margin
    
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EIPC: Expected Improvement per unit cost
    eipc = ei / cost_safe
    
    # Dynamic threshold: reject candidates with low EIPC
    # Estimate budget progress from f_best (lower f_best = later in search)
    # The minimum of powell 4d is 0, so f_best approaches 0 as we progress
    # Use f_best to estimate how much budget remains
    
    # Heuristic: if f_best is close to 0, we're late; if far from 0, we're early
    # The scale of powell 4d values: at random points, values can be ~1-10
    # At the minimum, value is 0
    
    # Progress estimate: f_best / (f_best + 1) gives 0 when f_best=0, 1 when f_best is large
    progress = f_best / (abs(f_best) + 1.0)
    # progress ~ 1 early, ~ 0 late
    
    # Budget-aware: early on, we can afford to explore; late on, exploit
    # The threshold should be higher (more selective) when budget is tight
    # Estimate remaining budget: total budget is 30, 5 initial points cost ~7.5 (avg cost 1.5)
    # So ~22.5 remaining after initial points
    
    # Dynamic threshold: use percentile-based selection
    # Early: select top 30% by EIPC; Late: select top 10%
    # Interpolate selection fraction based on progress
    sel_frac_early = 0.3
    sel_frac_late = 0.1
    sel_frac = sel_frac_late + (sel_frac_early - sel_frac_late) * progress
    
    # Compute threshold as the (1 - sel_frac) quantile of EIPC
    # Only candidates above this threshold get a positive score
    threshold = np.percentile(eipc, 100 * (1 - sel_frac))
    
    # Score: EIPC for selected candidates, 0 for rejected
    # But we need to return scores for all 2000 candidates
    # Selected candidates get their EIPC value; rejected get a small negative value
    # to ensure they are never chosen
    
    # Apply threshold: candidates above threshold get EIPC, others get a penalty
    # Use a smooth transition rather than hard cutoff to avoid discontinuities
    # But the directive says "discrete" so use a hard cutoff
    
    score = np.where(eipc >= threshold, eipc, -1e6)
    
    # Add a small UCB term for exploration among selected candidates
    # This helps break ties and encourages diversity in the selected set
    # UCB for minimization: -mu + beta * sigma
    # Scale by 1/cost to prefer cheap exploratory points
    
    # Adaptive beta: larger when progress is high (early), smaller when low (late)
    beta = 0.3 + 0.4 * progress
    
    # Only add UCB term to selected candidates
    ucb_term = (-mu + beta * sigma) / cost_safe
    score = np.where(eipc >= threshold, score + 0.1 * ucb_term, score)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Deadline check: if we're running out of time, return immediately
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
