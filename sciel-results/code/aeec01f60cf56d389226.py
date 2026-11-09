"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track state across calls
_state = {
    'start_time': None,
    'call_count': 0,
    'f_best_history': []
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Dynamic weighted combination of Expected Improvement
    and predictive uncertainty (Pareto-based multi-objective approach).
    
    The weight on exploration increases as we have fewer evaluations (early stages)
    and decreases as we approach the budget (later stages), balancing exploration
    and exploitation more effectively than a fixed beta LCB.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Initialize state on first call
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    _state['call_count'] += 1
    
    # Deadline guard: 330s wall-clock limit, use 325s as safety margin
    elapsed = time.time() - _state['start_time']
    if elapsed > 325:
        # Return EI-only scores quickly to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    n = len(mu)
    
    # Track f_best history to estimate progress
    _state['f_best_history'].append(f_best)
    
    # Total budget is 30 evaluations (5 initial + 25 more)
    # call_count starts at 1 for the first acquisition call (after 5 initial points)
    # So we've done 5 + (call_count - 1) = call_count + 4 evaluations so far
    evals_done = _state['call_count'] + 4
    total_evals = 30
    progress = min(evals_done / total_evals, 1.0)
    
    # Dynamic weight: start with more exploration, shift to exploitation
    # At progress=0 (early), weight_explore ~ 0.5
    # At progress=1 (late), weight_exploit dominates
    # Use a smooth transition
    beta = 0.1 + 0.5 * progress  # beta increases from 0.1 to 0.6
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for proper weighting
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # Normalize uncertainty to [0, 1] range
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-12:
        sigma_norm = np.zeros_like(sigma_safe)
    else:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Weighted combination: exploit + explore
    # As progress increases, we weight EI more heavily
    w_ei = 1.0 - 0.3 * (1.0 - progress)  # w_ei goes from 0.7 to 1.0
    w_sigma = 0.3 * (1.0 - progress)     # w_sigma goes from 0.3 to 0.0
    
    scores = w_ei * ei_norm + w_sigma * sigma_norm
    
    # Scale back to original EI scale for meaningful comparison
    scores = scores * ei_range + ei_min
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
