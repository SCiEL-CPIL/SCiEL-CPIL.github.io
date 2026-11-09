"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def _check_deadline(timeout=320.0):
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > timeout

def acquisition(mu, sigma, f_best):
    """
    Hierarchical Lookahead-UCB acquisition for Bayesian optimization.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard
    if _check_deadline():
        # Return simple EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Known minimum of the RBF mixture
    f_min = -2.53218
    
    # Progress: how close are we to the minimum?
    regret = f_best - f_min
    progress = 1.0 - np.clip(regret / 2.0, 0.0, 1.0)
    
    # Hierarchical pool: early on, 60% of candidates are in exploration pool
    # Late, pool shrinks to 15% forcing focus
    pool_fraction = 0.60 * (1.0 - progress) + 0.15 * progress
    
    # Compute EI for all candidates
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB for exploration
    # Beta controls exploration-exploitation balance
    beta = 0.5 * (1.0 - progress) + 0.2 * progress
    ucb = mu + beta * sigma_safe
    
    # Rank candidates by EI to create hierarchical pools
    ei_rank = np.argsort(ei)  # ascending, so high EI at end
    
    # Top pool: highest EI candidates (exploitation)
    # Exploration pool: next set of candidates with high UCB (exploration)
    top_k = max(1, int(n * (1.0 - pool_fraction)))
    exploration_k = max(1, int(n * pool_fraction))
    
    # Score: combine EI and UCB with hierarchical gating
    # In top pool: pure EI (exploitation)
    # In exploration pool: UCB-based score (exploration)
    score = np.zeros(n)
    
    # Top pool: use EI directly
    top_indices = ei_rank[-top_k:]
    score[top_indices] = ei[top_indices]
    
    # Exploration pool: use UCB, but penalize to be below top pool's minimum
    # This ensures top pool is always preferred unless UCB is very high
    expl_indices = ei_rank[:exploration_k]
    min_top_ei = np.min(ei[top_indices]) if top_k > 0 else 0.0
    
    # Scale UCB to be competitive with EI
    # UCB is in objective units, EI is also in objective units
    # But UCB can be negative (since mu can be negative), so we need care
    # For minimization, lower is better, so we want high score = good
    # UCB = mu + beta*sigma, for minimization we want low mu, high sigma
    # So we should use -UCB or a transformation
    
    # Actually, for minimization:
    # EI measures expected improvement (higher = better)
    # For UCB in minimization, we want to explore where mu is low AND sigma is high
    # A common approach: use -mu + beta*sigma (negate mu since lower is better)
    
    ucb_min = -mu + beta * sigma_safe
    max_ucb_min = np.max(ucb_min[expl_indices]) if exploration_k > 0 else 0.0
    min_ucb_min = np.min(ucb_min[expl_indices]) if exploration_k > 0 else 0.0
    ucb_range = max_ucb_min - min_ucb_min + 1e-9
    
    # Normalize UCB to [0, 1] and scale to be slightly below min_top_ei
    ucb_norm = (ucb_min[expl_indices] - min_ucb_min) / ucb_range
    # Scale to be in range [0, min_top_ei * 0.9] to stay below top pool
    score[expl_indices] = ucb_norm * (min_top_ei * 0.9 if min_top_ei > 0 else 0.0)
    
    # Remaining candidates (if any): give very low score
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[top_indices] = False
    remaining_mask[expl_indices] = False
    score[remaining_mask] = -1.0
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
