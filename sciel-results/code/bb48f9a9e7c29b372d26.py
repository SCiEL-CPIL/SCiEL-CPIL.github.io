import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Hypervolume-inspired scoring for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates, higher is better.
    """
    # Track time to ensure we finish well within the 300s limit
    t_start = time.time()
    deadline = t_start + 250.0  # Safety margin
    
    # Handle potential numerical issues
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Normalize mu and sigma to [0, 1] range for consistent scoring
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    mu_norm = (mu - mu_min) / mu_range  # 0 = best (lowest mu), 1 = worst
    
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    sigma_norm = (sigma_safe - sigma_min) / sigma_range  # 0 = least uncertain, 1 = most uncertain
    
    # For minimization: we want LOW mu and HIGH sigma (uncertainty)
    # Create desirability scores where:
    #   - Low mu_norm (close to 0) is good
    #   - High sigma_norm (close to 1) is good
    
    # Base exploitation score: higher when mu is lower
    exploit_score = 1.0 - mu_norm
    
    # Exploration score: higher when sigma is higher
    explore_score = sigma_norm
    
    # Hypervolume-inspired combination:
    # We want to reward points that improve the Pareto front in (mu, sigma) space.
    # A point is valuable if it has low mu AND/OR high sigma in a way that
    # adds volume to the non-dominated set.
    #
    # Use a smooth approximation: the contribution of a point to hypervolume
    # is roughly proportional to the area it "claims" in the objective space.
    # For a single point (mu_i, sigma_i), its contribution depends on how many
    # other points it dominates or is dominated by.
    #
    # Approximate approach: use a weighted geometric mean that rewards
    # balanced improvement in both objectives, with emphasis on exploitation.
    #
    # score = (exploit_score)^alpha * (explore_score)^beta
    # This is a smooth, non-linear trade-off that naturally favors points
    # that are good in at least one dimension and not terrible in the other.
    
    alpha = 0.6  # Weight for exploitation (low mu)
    beta = 0.4   # Weight for exploration (high sigma)
    
    # Clip to avoid log(0) issues
    exploit_clipped = np.clip(exploit_score, 1e-8, 1.0)
    explore_clipped = np.clip(explore_score, 1e-8, 1.0)
    
    # Geometric mean combination
    scores = (exploit_clipped ** alpha) * (explore_clipped ** beta)
    
    # Add a bonus for points that are in the "sweet spot":
    # reasonably low mu and moderately high sigma
    # This helps avoid pure exploitation or pure exploration extremes
    sweet_spot_bonus = 0.15 * exploit_score * explore_score * (1.0 - np.abs(exploit_score - explore_score))
    scores += sweet_spot_bonus
    
    # Scale scores to be clearly positive and well-separated
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    scores_range = scores_max - scores_min if scores_max > scores_min else 1.0
    scores = (scores - scores_min) / scores_range  # Normalize to [0, 1]
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline - if we're running out of time, return immediately
    if time.time() > deadline:
        return scores
    
    return scores
