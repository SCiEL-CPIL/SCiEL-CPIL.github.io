"""Acquisition function: Time-Adaptive UCB/EI for ST1D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_global_start_time = None

def _ei_scores(mu, sigma, f_best):
    """Compute Expected Improvement scores (higher is better) for minimization."""
    eps = 1e-9
    # Avoid division by zero if sigma is 0
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # Standard EI formula for minimization:
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _ucb_scores(mu, sigma, beta):
    """Compute UCB scores for minimization (higher is better)."""
    # We want to minimize f(x), so we look for low mu.
    # UCB for minimization: -mu + beta * sigma
    # Higher score means "better" candidate (lower predicted f, higher uncertainty).
    return -mu + beta * sigma

def acquisition(mu, sigma, f_best):
    global _global_start_time
    
    n = len(mu)
    
    # Initialize global start time
    if _global_start_time is None:
        _global_start_time = time.time()
        
    elapsed = time.time() - _global_start_time
    
    # Deadline guard: 330s total budget.
    if elapsed > 320.0:
        # Fallback: fast EI
        return _ei_scores(mu, sigma, f_best)

    # Estimate current step
    # The problem states 30 evaluations total.
    # We assume the calls are distributed somewhat evenly, but we can just use elapsed time
    # as a proxy for progress if we know the approximate time per step.
    # However, a more robust proxy is just the elapsed time relative to a typical total duration.
    # Let's assume the whole process takes up to ~300s (though it might be faster).
    # A safer heuristic: 
    # If we have just started, we are in exploration.
    # If we are near the end, we are in exploitation.
    
    # Let's define a progress variable t in [0, 1]
    # We don't know the exact step count directly, but we can infer it from elapsed time.
    # Average time per step = elapsed / steps_so_far? No, we don't know steps_so_far.
    
    # Alternative: Use a fixed schedule based on elapsed time.
    # Most BO loops are fast (milliseconds to seconds per step).
    # Let's assume the entire 30-step process should finish well within 300s.
    # If elapsed is small, t is small. If elapsed is large, t is large.
    # Let's map elapsed time to [0, 1] over a window of, say, 100 seconds.
    # If it takes longer, we cap it at 1.0 (full exploitation).
    # This is robust because if the evaluator is slow, we exploit more.
    
    time_window = 100.0
    t = np.clip(elapsed / time_window, 0.0, 1.0)
    
    # We want to blend between UCB (exploration) and EI (exploitation).
    # At t=0 (start), weight on UCB is high, EI is low.
    # At t=1 (end), weight on UCB is low, EI is high.
    
    # Let's use a smooth transition.
    # w_ucb = 1 - t
    # w_ei = t
    
    # However, simple weighted sum of raw scores is problematic because EI and UCB have different scales.
    # EI is bounded by f_best (improvement). UCB can be large negative or positive.
    # It is better to use a switch or a rank-based blend, or normalize scores.
    
    # Let's try a different structural approach:
    # Use UCB with a decaying beta.
    # beta(t) = beta_max * (1 - t)^2 + beta_min
    # This naturally shifts from exploration to exploitation.
    # We can then just return the UCB score with this adaptive beta.
    # But UCB is not as precise as EI for final refinement.
    
    # Better: Return EI when t is high, and UCB when t is low.
    # To make it smooth, we can compute both and pick the one with higher "confidence" 
    # or just blend their normalized ranks as before, but with time-varying weights.
    
    # Let's stick to the rank-based blend but make the weights time-dependent.
    # w_ei = t
    # w_ucb = 1 - t
    
    # Early: w_ucb ~ 1, w_ei ~ 0 -> Pure UCB exploration.
    # Late: w_ucb ~ 0, w_ei ~ 1 -> Pure EI exploitation.
    
    # Compute EI
    ei = _ei_scores(mu, sigma, f_best)
    
    # Compute UCB with a reasonable fixed beta for the rank calculation.
    # The rank of UCB depends on beta. Let's use beta=2.0 for a strong exploration signal.
    ucb = _ucb_scores(mu, sigma, 2.0)
    
    # Compute ranks (1 is best/highest score, n is worst/lowest score)
    # We want higher score = better.
    # np.argsort(np.argsort(x)) gives 0-indexed ranks where 0 is smallest.
    # We want 1 to be best (largest value).
    # So we reverse the order: rank = n - rank_0indexed.
    
    ei_rank = np.empty(n)
    ucb_rank = np.empty(n)
    
    # Handle potential NaNs/Infs in mu/sigma just in case
    ei_safe = np.where(np.isfinite(ei), ei, -np.inf)
    ucb_safe = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    ei_order = np.argsort(ei_safe)
    ucb_order = np.argsort(ucb_safe)
    
    ei_ranks = np.empty(n)
    ucb_ranks = np.empty(n)
    ei_ranks[ei_order] = np.arange(n)
    ucb_ranks[ucb_order] = np.arange(n)
    
    # Invert ranks so that 0 is the best (max value)
    ei_rank_inv = n - 1 - ei_ranks
    ucb_rank_inv = n - 1 - ucb_ranks
    
    # Blend
    w_ei = t
    w_ucb = 1.0 - t
    
    blended_rank = w_ei * ei_rank_inv + w_ucb * ucb_rank_inv
    
    # Convert back to scores (higher is better)
    # We can just use the blended rank as the score, or map it to a range.
    # Using the rank directly is fine as long as it's monotonic.
    scores = blended_rank
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
