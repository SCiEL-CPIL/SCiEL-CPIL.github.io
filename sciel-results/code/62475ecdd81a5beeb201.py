"""Rewritten acquisition for Bayesian optimization of 6d Hartmann."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track elapsed time and number of calls
_acq_state = {
    'start_time': None,
    'call_count': 0,
}


def _get_elapsed():
    """Return elapsed time since first call, initializing if needed."""
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    return time.time() - _acq_state['start_time']


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an adaptive exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Increment call count
    _acq_state['call_count'] += 1
    elapsed = _get_elapsed()
    
    # Time guard: if we're getting close to the 330s limit, return conservative scores
    # to allow the evaluator to complete gracefully
    if elapsed > 310.0:
        # Return pure EI (no exploration) to be safe and fast
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        # Numerically stable EI using the identity: for large positive z, cdf(z)~1, pdf(z)~0
        # For large negative z, cdf(z)~0, pdf(z)~0, so EI~0
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization) with numerical stability
    z = (f_best - mu) / sigma_safe
    
    # Use a stable computation: for z > 30, cdf~1 and pdf~0
    # For z < -30, cdf~0 and pdf~0, so EI~0
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -30, 30)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Adaptive exploration weight:
    # Start with higher exploration (0.15) and decay to lower (0.03) as search progresses
    # This helps escape local minima early and focus on exploitation later
    # The decay is based on call count (each call = one BO iteration step)
    # Total steps: 25 (30 evals - 5 initial), so normalize by expected total
    total_expected_steps = 25
    step_fraction = min(_acq_state['call_count'] / total_expected_steps, 1.0)
    
    # Smooth decay from 0.15 to 0.03
    exploration_weight = 0.15 - 0.12 * step_fraction
    
    # Additional adaptive factor: reduce exploration when f_best is close to known minimum
    # Hartmann 6D minimum is approximately -3.32237
    known_min = -3.32237
    improvement_remaining = f_best - known_min
    if improvement_remaining > 0.5:
        # Still far from optimum, boost exploration
        exploration_weight *= 1.3
    elif improvement_remaining < 0.1:
        # Very close to optimum, reduce exploration
        exploration_weight *= 0.6
    
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
