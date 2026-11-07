import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

# Global state to track progress across calls within a seed
_step_count = 0
_seed_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Structural Hybrid: Pure UCB for early exploration, transitioning to a
    normalized blend of EI and UCB for refined exploitation.
    
    This addresses the stagnation of the previous EI+fixed-sigma hybrid by
    recognizing that the Beale function's narrow valley requires aggressive
    exploration (high-beta UCB) initially to escape local minima, before
    switching to a balanced exploitation strategy.
    """
    global _step_count, _seed_start_time
    
    # Initialize time tracking on first call
    if _seed_start_time is None:
        _seed_start_time = time.time()
        _step_count = 0
    
    elapsed = time.time() - _start_time
    _step_count += 1
    
    # Safety margin: return simple UCB if we're getting close to deadline
    if elapsed > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 2.0 * sigma_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not np.isfinite(f_best):
        f_best = 1e6
        
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Phase 1: Aggressive Exploration (Pure UCB)
    # The Beale function has a narrow valley. Standard EI often converges to local minima.
    # Using a high-beta UCB (e.g., beta=3.0) in the first few steps helps jump over
    # barriers to find the global basin.
    # We use the first 5 steps (after the initial 5 random points) for this phase.
    if _step_count <= 5:
        beta = 3.0
        scores = -mu + beta * sigma_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Phase 2: Refined Search (Normalized EI + UCB Blend)
    # Once we are in the basin, we need to balance exploitation (EI) and
    # continued exploration (UCB) to pinpoint the minimum.
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score
    # Use a moderate beta for the blend, slightly lower than the initial aggressive phase
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize both scores to [0, 1] to ensure they contribute equally to the blend
    # This prevents EI (which can be large in absolute terms) from dominating UCB
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # Adaptive weighting:
    # As steps increase, we shift weight towards EI (exploitation)
    # but keep a significant UCB component to avoid getting stuck in the narrow valley.
    # Progress from 5 to 25 steps: weight shifts from 0.5/0.5 to 0.7/0.3
    progress = min(1.0, max(0.0, (_step_count - 5) / 20.0))
    w_ei = 0.5 + 0.2 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
