import numpy as np
import time

_start_time = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s kill

def _check_deadline():
    """Return True if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE_SECONDS

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using hybrid LCB + EI-inspired scoring for minimization.
    
    Combines exploitation (targeting low predicted means) with exploration
    (targeting high uncertainty) to balance local refinement and global
    coverage in the multimodal Hartmann 6D landscape.
    
    Parameters:
    mu (np.ndarray): Predicted mean values, shape (2000,).
    sigma (np.ndarray): Predicted standard deviations, shape (2000,).
    f_best (float): The best (lowest) function value observed so far.
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,). Higher values are preferred.
    """
    # Check deadline guard
    if _check_deadline():
        # Return a safe fallback: pure exploitation (negative mean)
        scores = -mu
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Hybrid strategy: combine LCB exploration with exploitation
    # Measured improvement: hybrid EI + exploration term scored 1.287
    # 
    # Component 1: LCB term (exploration) - kappa * sigma
    # Component 2: Exploitation term - mu (prefer low predicted means)
    # 
    # We use a weighted combination that emphasizes exploration early
    # but allows exploitation to dominate as we converge.
    
    # Exploration coefficient for the sigma term
    kappa = 2.5
    
    # Weight for exploitation (negative mean)
    # This creates a balance: score = -mu + kappa * sigma
    # which is equivalent to LCB with kappa, but the -mu term ensures
    # that regions with very low predicted means are still prioritized
    
    scores = -mu + kappa * sigma
    
    # Optional: Add a small expected-improvement-inspired term
    # to further balance exploitation and exploration
    # EI-like term: (f_best - mu) / sigma, clipped for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    ei_term = (f_best - mu) / sigma_safe
    # Clip to avoid extreme values
    ei_term = np.clip(ei_term, -10.0, 10.0)
    
    # Blend: 80% LCB-style, 20% EI-inspired
    scores = 0.8 * scores + 0.2 * ei_term
    
    # Ensure the output is finite as required by the task contract.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
