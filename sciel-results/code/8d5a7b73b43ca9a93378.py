import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pareto front selection in (mu, sigma) space.
    
    Identifies non-dominated candidates where lower mu and higher sigma are both desirable.
    Scores are based on convex hull distance to the Pareto front, forcing simultaneous
    pursuit of exploitation (low mean) and exploration (high variance).
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Step 1: Identify Pareto non-dominated points in (mu, sigma) space ---
    # We want to minimize mu and maximize sigma.
    # A point is non-dominated if no other point has both lower mu and higher sigma.
    
    # Sort by mu ascending, then by sigma descending
    sort_idx = np.lexsort((sigma_safe[::-1], mu))
    mu_sorted = mu[sort_idx]
    sigma_sorted = sigma_safe[sort_idx]
    
    # Find non-dominated points
    # A point is non-dominated if its sigma is greater than all subsequent points' sigma
    # (since we sorted by mu ascending, subsequent points have higher or equal mu)
    max_sigma_suffix = np.zeros(n)
    max_sigma_suffix[-1] = sigma_sorted[-1]
    for i in range(n - 2, -1, -1):
        max_sigma_suffix[i] = max(sigma_sorted[i], max_sigma_suffix[i + 1])
    
    # Non-dominated: sigma[i] == max_sigma_suffix[i] (i.e., no later point has higher sigma)
    is_non_dominated = (sigma_sorted == max_sigma_suffix)
    
    # Get indices of non-dominated points in original ordering
    non_dom_idx_sorted = sort_idx[is_non_dominated]
    
    # If too few or too many non-dominated points, adjust
    if len(non_dom_idx_sorted) < 2:
        # Fallback: just use top few by a combined score
        combined = -mu + sigma_safe
        top_k = np.argsort(combined)[-10:]
        non_dom_idx_sorted = top_k
    elif len(non_dom_idx_sorted) > 50:
        # Keep only the most extreme points: lowest mu and highest sigma
        # Take the first 25 by mu (lowest) and last 25 by sigma (highest)
        by_mu = non_dom_idx_sorted[:25]
        by_sigma = non_dom_idx_sorted[-25:]
        non_dom_idx_sorted = np.unique(np.concatenate([by_mu, by_sigma]))
    
    # --- Step 2: Compute scores based on distance to Pareto front ---
    # For each candidate, compute a score that rewards being close to the Pareto front
    # in a way that balances exploitation and exploration.
    
    # Normalize mu and sigma to [0, 1] for consistent scaling
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Pareto front points in normalized space
    mu_nd_norm = (mu[non_dom_idx_sorted] - mu_min) / mu_range
    sigma_nd_norm = (sigma_safe[non_dom_idx_sorted] - sigma_min) / sigma_range
    
    # For each candidate, compute the minimum "dominance distance" to any Pareto point
    # Distance metric: we want candidates that are close to the front in a way that
    # balances low mu and high sigma. Use a weighted Euclidean distance in normalized space.
    
    # Compute distances from each candidate to each Pareto point
    # To keep this efficient, we'll compute a scalar score for each candidate
    # based on how well it aligns with the Pareto front.
    
    # Alternative approach: score each candidate by how "Pareto-like" it is.
    # A candidate is good if it has low mu and/or high sigma relative to the front.
    
    # Compute the "front score" for each candidate:
    # - If the candidate is on or above the Pareto front (in the sense of having
    #   higher sigma for its mu, or lower mu for its sigma), it gets a high score.
    # - Otherwise, it gets a score based on how close it is to the front.
    
    # For each candidate, find the Pareto point with the closest mu
    # and check if the candidate's sigma is above or below that point's sigma.
    
    # Efficient approach: for each candidate, compute a "violation" score.
    # A candidate violates the Pareto front if there exists a Pareto point with
    # both lower mu and higher sigma.
    
    # For each candidate, compute the maximum sigma among Pareto points with mu <= candidate's mu
    # If candidate's sigma is less than this, it is dominated.
    
    # Build a lookup: for each candidate, find the max sigma of Pareto points with mu <= mu_candidate
    # We can do this by sorting and using cumulative max.
    
    # Sort Pareto points by mu
    pareto_sort_idx = np.argsort(mu_nd_norm)
    mu_pareto_sorted = mu_nd_norm[pareto_sort_idx]
    sigma_pareto_sorted = sigma_nd_norm[pareto_sort_idx]
    
    # For each candidate, find where it falls in the sorted Pareto mu array
    # and get the cumulative max sigma up to that point
    cum_max_sigma_pareto = np.maximum.accumulate(sigma_pareto_sorted)
    
    # For each candidate, find the rightmost Pareto point with mu <= candidate's mu
    # Using searchsorted
    candidate_mu_norm = mu_norm
    insert_pos = np.searchsorted(mu_pareto_sorted, candidate_mu_norm, side='right')
    
    # For candidates with insert_pos == 0, no Pareto point has mu <= candidate's mu
    # For others, the relevant Pareto point is at insert_pos - 1
    valid_mask = insert_pos > 0
    max_sigma_for_mu = np.zeros(n)
    max_sigma_for_mu[valid_mask] = cum_max_sigma_pareto[insert_pos[valid_mask] - 1]
    
    # Candidates that are dominated: their sigma is less than max_sigma_for_mu
    # The "violation" is how much below the front they are
    violation = np.maximum(0, max_sigma_for_mu - sigma_norm)
    
    # Score: higher is better. We want to reward low violation and also reward
    # being close to the front in a balanced way.
    
    # Base score: negative violation (so lower violation = higher score)
    # Plus a bonus for having low mu and high sigma (balanced exploration/exploitation)
    
    # Normalize violation to [0, 1]
    max_violation = np.max(violation) if np.max(violation) > 0 else 1.0
    violation_norm = violation / max_violation
    
    # Score components:
    # 1. Low violation (close to or on the Pareto front)
    front_score = 1.0 - violation_norm
    
    # 2. Balanced score: reward low mu and high sigma
    # Use a weighted combination
    exploit_score = 1.0 - mu_norm  # Low mu is good
    explore_score = sigma_norm  # High sigma is good
    
    # Combine with weights that emphasize balance
    balance_score = 0.5 * exploit_score + 0.5 * explore_score
    
    # Final score: combine front score and balance score
    # Points on the Pareto front with good balance get the highest scores
    scores = 0.6 * front_score + 0.4 * balance_score
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
