"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Predictive Entropy Search (PES) variant for Bayesian optimization.
    Scores candidates by expected reduction in the posterior entropy of the
    global minimum, approximated via mutual information between the function
    value at the candidate and the location/value of the current best.
    """
    t_start = time.time()
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Normalize sigma for consistent scaling
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = sigma_safe
    
    # Normalize mu relative to f_best for consistent scaling
    mu_range = np.max(mu_safe) - np.min(mu_safe)
    if mu_range > 0:
        mu_norm = (mu_safe - np.min(mu_safe)) / mu_range
    else:
        mu_norm = np.zeros_like(mu_safe)
    
    # PES-like score: combines exploitation (low mu) with exploration weighted
    # by uncertainty about the global minimum.
    # The key insight: candidates with high sigma are valuable because they
    # could reveal a new global minimum, reducing entropy about where the
    # true minimum is.
    
    # Compute a "global minimum uncertainty" score:
    # 1. Candidates far below f_best in mean are likely to improve the best
    # 2. Candidates with high sigma could potentially be much lower than f_best
    # 3. Combine these to target regions that could shift the global minimum
    
    # Exploitation component: how much lower is the predicted mean than f_best?
    improvement = f_best - mu_safe  # Positive if mu is below f_best
    
    # Exploration component: probability that f(x) < f_best, approximated
    # by the ratio of sigma to the gap, scaled by a sensitivity factor
    gap = np.maximum(f_best - mu_safe, 1e-9)
    # Z-score-like measure: how many sigmas below f_best is the mean?
    z_score = (f_best - mu_safe) / sigma_safe
    # Probability that f(x) < f_best, using normal CDF approximation
    # For large positive z, prob ~ 1; for large negative z, prob ~ 0
    # Use a smooth approximation: sigmoid(0.5 * z_score)
    prob_improve = 1.0 / (1.0 + np.exp(-0.5 * z_score))
    
    # Expected improvement-like term
    ei_term = improvement * prob_improve + sigma_safe * (1 - prob_improve)
    
    # PES entropy reduction: candidates that are uncertain AND could be
    # significantly below f_best reduce the most entropy about the global min
    # Weight by how "informative" the candidate is about the global structure
    # High sigma candidates far from current best are most informative
    
    # Combine: use a weighted blend of EI and a diversity/informativeness term
    # The informativeness is high when sigma is large and the candidate is
    # in a region not well-explored (high posterior uncertainty)
    informativeness = sigma_norm * (1.0 + 0.5 * np.abs(z_score))
    
    # Final PES-inspired score
    # Balance between direct improvement and entropy reduction
    alpha = 0.6  # Weight for EI-like term
    beta = 0.4   # Weight for entropy reduction / informativeness
    
    scores = alpha * ei_term + beta * informativeness
    
    # Add a small exploitation boost for points very close to f_best
    # to ensure we don't miss obvious improvements
    close_to_best = np.exp(-np.abs(improvement) / (sigma_safe + 1e-9))
    scores += 0.1 * close_to_best * np.maximum(improvement, 0)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard: if we're taking too long, return what we have
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 330s kill
        return scores
    
    return scores
# EVOLVE-BLOCK-END
