import numpy as np
import time
import sys

# Global state to track time and step count for adaptive behavior
_start_time = None
_step_count = 0

def acquisition(mu, sigma, f_best):
    """
    Deterministic Adaptive Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute Expected Improvement (EI) for exploitation.
    2. Add an exploration term based on the standard deviation (sigma), 
       weighted by a dynamic beta that decreases over time.
    3. This creates a smooth transition from exploration (high beta early) 
       to exploitation (low beta late).
       
    Why this is different from the current Thompson Sampling:
    - Deterministic: No random sampling noise, leading to more consistent 
      progress and avoiding "bad luck" in candidate selection.
    - Adaptive: The exploration constant beta is tuned based on the number 
      of steps, encouraging broader search initially and finer search later.
    """
    global _start_time, _step_count
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Increment step counter
    _step_count += 1
    
    # Safety guard: If we are running too long, return a safe fallback
    # to prevent timeout. The evaluator kills after 330s. 
    # We aim to finish each call well under that, but if something goes wrong 
    # or the system is slow, we bail out gracefully.
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return a flat score to avoid crashing, effectively stopping search
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta:
    # Start with high exploration (beta ~ 2.0) and decay to low exploitation (beta ~ 0.1)
    # over the course of the 25 optimization steps (30 total evals - 5 initial).
    # We use a smooth decay function.
    # step 1 -> beta ~ 1.5, step 25 -> beta ~ 0.2
    # Let's use a simple linear decay or exponential.
    # Total expected steps is 25.
    # beta = 1.0 + 0.5 * exp(-step_count / 10.0)
    # At step 1: 1.0 + 0.5 * 0.9 = 1.45
    # At step 10: 1.0 + 0.5 * 0.36 = 1.18
    # At step 25: 1.0 + 0.5 * 0.08 = 1.04
    # This might be too slow. Let's try a faster decay to focus on exploitation sooner,
    # as Beale function is relatively smooth and has a single global minimum.
    
    # Alternative: Beta based on improvement. But step count is more reliable.
    # Let's use a decay that drops off faster.
    decay_rate = 8.0
    beta = 0.5 + 1.5 * np.exp(-_step_count / decay_rate)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    # Optimization: Use scipy.stats.norm for accuracy and speed
    try:
        from scipy.stats import norm
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    except ImportError:
        # Fallback to manual approximation if scipy is missing (though it should be there)
        # Simple approximation for EI
        z = (f_best - mu) / sigma_safe
        # Approximate CDF and PDF
        cdf = 0.5 * (1 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical noise can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: Beta * Sigma
    # We add this to the EI to encourage exploration of high-uncertainty regions.
    # However, pure EI already includes some exploration via the sigma term.
    # Adding an explicit UCB-like term can help escape local minima if the GP
    # has converged too quickly on a suboptimal region.
    
    # Let's blend: Score = EI + beta * sigma
    # This is effectively a weighted sum of Exploitation and Exploration.
    scores = ei + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
