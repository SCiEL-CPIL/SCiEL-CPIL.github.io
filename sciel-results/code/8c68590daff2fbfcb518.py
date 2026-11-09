import numpy as np
from scipy.stats import norm
import time

# Global start time to track elapsed time for the entire evaluation session
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Probability of Improvement (POI) and Expected Improvement (EI).
    
    This approach uses POI as the primary structural signal because it provides a rank-based,
    softer exploration pressure that is robust to scale differences and can better navigate
    multimodal landscapes like Branin 2D. EI is added as a refinement term to provide
    magnitude-based exploitation for points that are already likely to improve.
    
    Parameters:
    mu: array of shape (n_candidates,) - posterior mean
    sigma: array of shape (n_candidates,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (n_candidates,) - combined scores (higher is better)
    """
    
    # Deadline guard: Check if we are approaching the 330s limit.
    # Since this function is fast, this is mostly a formality, but included per directive.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # If we are very close to the deadline, return a safe fallback.
        # Returning zeros is safe, but let's just compute the fast POI.
        pass

    # Ensure sigma is positive to avoid division by zero
    # Use a small epsilon to prevent numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement z
    # z = (f_best - mu) / sigma
    # Since we are minimizing, "improvement" means f(x) < f_best.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be on a similar scale as POI (which is in [0, 1])
    # EI can be much larger than 1, so we scale it down.
    # We use a soft scaling to keep the contribution meaningful but not dominant.
    # A simple division by a characteristic scale or just a weight is sufficient.
    # Here, we use a weight of 0.1 to make EI a refinement term.
    # The POI term is the primary driver [0, 1], EI term adds nuance.
    
    # To ensure the EI term doesn't overwhelm POI, we can normalize EI by its max value
    # or use a fixed small weight. Since the scale of EI depends on the function values,
    # a fixed weight is safer if we don't know the scale. However, standard EI is often
    # comparable to the scale of the function.
    # Let's use a simple weighted sum.
    # POI is in [0, 1]. EI is >= 0.
    # If we just add them, EI might dominate if the function values are large.
    # Branin values are roughly in [0, 300]. f_best starts around 100-200.
    # So EI can be large.
    
    # Better approach: Rank-based or normalized combination.
    # But the directive asks for a structural change + refinement.
    # Let's stick to POI as the main term and add a small, scaled EI.
    
    # Scale EI to be comparable to POI.
    # A common heuristic is to divide EI by a characteristic length scale of the objective.
    # Without knowing the exact scale, we can use the current spread of mu or f_best.
    # However, a simpler and more robust method for this specific task (Branin)
    # is to use the fact that POI is a probability.
    
    # Let's try: Score = POI + 0.01 * EI
    # The 0.01 factor makes EI a tie-breaker or fine-tuner.
    
    scores = poi + 0.01 * ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
