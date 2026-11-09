import numpy as np
from scipy.stats import norm
import time

# Global state for tracking
_state = {
    'start_time': None,
    'n_calls': 0,
    'f_best_history': []
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    Instead of independent EI/cost, we use a batch-aware scoring that
    accounts for redundancy among candidates by penalizing clusters
    of similar predicted improvement, plus adaptive cost sensitivity.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = prefer)
    """
    global _state
    
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['n_calls'] += 1
    _state['f_best_history'].append(f_best)
    
    elapsed = time.time() - _state['start_time']
    
    # Safety: if we're close to timeout, use a fast fallback
    if elapsed > 290:
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        cost_safe = np.where(cost > 1e-9, cost, 1e-9)
        z = np.clip((f_best - mu) / sigma_safe, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    n = len(mu)
    n_calls = _state['n_calls']
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    phi = norm.pdf(z_clipped)
    cdf = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * cdf + sigma_safe * phi
    
    # Normalize EI to [0, 1] range for better scaling
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei.copy()
    
    # Normalize cost to [0, 1] range
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    if cost_max - cost_min > 1e-9:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost_safe)
    
    # Adaptive parameters based on progress
    # Early: more exploration, later: more exploitation
    progress = min(n_calls / 25.0, 1.0)  # Assume ~25 calls total
    explore_weight = 1.0 - 0.5 * progress
    exploit_weight = 0.5 + 0.5 * progress
    
    # Cost sensitivity increases as budget depletes
    cost_sensitivity = 1.0 + 1.5 * progress
    
    # Base score: EI per unit cost, with adaptive cost sensitivity
    base_score = ei / (cost_safe ** cost_sensitivity)
    
    # Normalize base score
    base_max = np.max(base_score)
    if base_max > 1e-9:
        base_score_norm = base_score / base_max
    else:
        base_score_norm = base_score.copy()
    
    # Exploration bonus: prefer high uncertainty regions
    # But discount by cost (expensive uncertainty is less valuable)
    exploration = explore_weight * sigma_safe / (cost_safe + 0.5)
    expl_max = np.max(exploration)
    if expl_max > 1e-9:
        exploration_norm = exploration / expl_max
    else:
        exploration_norm = exploration.copy()
    
    # Redundancy penalty: penalize candidates that are similar to other
    # high-scoring candidates. We approximate this by looking at the
    # density of high-EI candidates in the candidate pool.
    # 
    # Instead of full multivariate computation, we use a simple approach:
    # For each candidate, compute how many other candidates have similar
    # (mu, sigma) profiles. This is expensive for 2000 points, so we
    # use a coarser approach: bin the candidates by their (mu, sigma)
    # values and penalize based on bin occupancy.
    
    # Create a combined feature vector for redundancy detection
    # Use normalized mu and sigma as features
    mu_norm = (mu - np.min(mu)) / (np.max(mu) - np.min(mu) + 1e-9)
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / (np.max(sigma_safe) - np.min(sigma_safe) + 1e-9)
    
    # Combine into a single score for clustering
    # High EI candidates that are clustered together should be penalized
    # We use a kernel density-like approach with a simple distance metric
    
    # For efficiency, use a grid-based density estimate
    # Divide the (mu_norm, sigma_norm) space into bins
    n_bins = 20
    mu_bin = np.clip((mu_norm * n_bins).astype(int), 0, n_bins - 1)
    sigma_bin = np.clip((sigma_norm * n_bins).astype(int), 0, n_bins - 1)
    
    # Count occupancy in each bin
    bin_counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i in range(n):
        bin_counts[mu_bin[i], sigma_bin[i]] += 1
    
    # Get the count for each candidate's bin
    candidate_bin_counts = bin_counts[mu_bin, sigma_bin]
    
    # Penalize candidates in crowded bins (high redundancy)
    # The penalty is proportional to the number of other candidates in the same bin
    # with high EI scores
    redundancy_penalty = candidate_bin_counts / n  # Normalize by total candidates
    
    # Weight the redundancy penalty by the EI of the candidate
    # Candidates with high EI in crowded regions get penalized more
    redundancy_adjusted = redundancy_penalty * ei_norm
    
    # Final score: combine base, exploration, and redundancy penalty
    scores = exploit_weight * base_score_norm + exploration_norm - 0.3 * redundancy_adjusted
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
