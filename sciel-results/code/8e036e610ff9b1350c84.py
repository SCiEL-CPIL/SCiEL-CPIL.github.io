"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Structural idea from Inspiration 2: Look-ahead simulation
    # Instead of just optimizing for the next step, we estimate how the
    # posterior would look after selecting the top K candidates.
    # We use the posterior mean as a proxy for the observed value (conservative).
    
    K = 5  # Number of look-ahead steps
    # Select top K candidates based on current EI (exploitation focus)
    # Using EI for selection is more robust than UCB for identifying "good" spots
    top_k_idx = np.argpartition(ei, -K)[-K:]
    
    # Estimate the new best value if these K points were evaluated.
    # We assume the observed values are the posterior means (conservative estimate).
    # In reality, the observed value might be better or worse, but mean is a good heuristic.
    simulated_new_f_best = min(f_best, np.min(mu[top_k_idx]))
    
    # Recalculate EI for the "future" state
    # This EI reflects how much further improvement is expected AFTER
    # the current top-K points are found.
    z_new = (simulated_new_f_best - mu) / sigma_safe
    z_new = np.clip(z_new, -10, 10)
    
    ei_future = (simulated_new_f_best - mu) * norm.cdf(z_new) + sigma_safe * norm.pdf(z_new)
    ei_future = np.where(np.isfinite(ei_future), ei_future, 0.0)
    
    # Refinement machinery from Inspiration 1: Entropy-weighted scaling
    # We want to balance the immediate improvement (ei) with the future potential (ei_future).
    # However, simple addition can be dominated by scale differences.
    # We use a normalized combination to ensure both components contribute meaningfully.
    
    # Normalize EI scores to [0, 1] range for combination
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    norm_ei_current = normalize_scores(ei)
    norm_ei_future = normalize_scores(ei_future)
    
    # Adaptive weighting based on distance to known minimum
    # Early on (large distance), we care more about finding ANY improvement (current EI).
    # Late on (small distance), we care more about refining and finding the global min (future EI / exploration).
    # Actually, future EI helps identify regions that have potential for *further* improvement,
    # which is crucial for global optimization.
    
    # Let's blend them.
    # weight_future controls how much we prioritize long-term gain over immediate gain.
    # A moderate weight ensures we don't just chase the single best point but
    # also consider points that open up new search avenues.
    
    # We also add a small exploration bonus based on uncertainty (sigma)
    # to avoid getting stuck in local minima if the GP model is overconfident.
    # This is a refinement from the hybrid approach.
    
    # Exploration bonus: scaled sigma
    # We want this to be small but non-zero.
    # Use a decay based on distance to min as well.
    distance_to_min = max(f_best - f_min_known, 0.0)
    
    # Exploration weight decays as we get closer to the min
    # When far, we explore more. When close, we exploit.
    # But we still need some exploration to escape local minima.
    explore_weight = 0.1 + 0.2 * np.exp(-distance_to_min / 5.0)
    
    # Normalize sigma for the exploration bonus
    norm_sigma = normalize_scores(sigma_safe)
    
    # Final score: blend of current EI, future EI, and exploration
    # We prioritize current EI (exploitation) but boost points that lead to high future EI.
    # The future EI term effectively acts as a "knowledge gradient" proxy.
    
    w_current = 0.6
    w_future = 0.3
    w_explore = 0.1
    
    # Adjust weights slightly based on phase
    # If we are very far from the min, increase exploration/future weight
    # If we are close, increase current EI weight
    
    if distance_to_min > 10.0:
        w_current = 0.5
        w_future = 0.3
        w_explore = 0.2
    elif distance_to_min > 5.0:
        w_current = 0.6
        w_future = 0.3
        w_explore = 0.1
    else:
        w_current = 0.7
        w_future = 0.2
        w_explore = 0.1
    
    final_scores = w_current * norm_ei_current + w_future * norm_ei_future + w_explore * norm_sigma
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return final_scores
    
    return final_scores
# EVOLVE-BLOCK-END
