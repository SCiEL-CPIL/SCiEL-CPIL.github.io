"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Set a deadline guard: we want to finish well within 300s
_DEADLINE = 280.0  # seconds
_start_time = time.time()


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural change: Blend cost-normalized Expected Improvement (EI) and UCB.
    # 
    # EI provides a principled measure of improvement, balancing exploitation
    # (via mu) and exploration (via sigma). UCB adds a stronger exploration
    # term, which is crucial for multimodal functions like Rastrigin.
    #
    # EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    # where Z = (f_best - mu(x)) / sigma(x)
    #
    # We compute EI and a UCB term, then combine them.
    # Both are normalized by the square root of the cost.
    
    # Calculate Z for EI
    # Avoid division by zero
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # For large negative Z, EI is 0. For large positive Z, EI is f_best - mu.
    # Using norm.cdf and norm.pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clamp EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # UCB term for additional exploration
    # Use a fixed beta for the UCB part, or adaptive?
    # The knowledge suggests beta=1.5 worked well.
    # Let's use a slightly adaptive beta to ensure early exploration.
    # progress: 1.0 (early) to 0.0 (late)
    f_max_estimate = 20.0
    progress = 1.0 - np.clip(f_best / f_max_estimate, 0.0, 1.0)
    
    # UCB beta: higher when progress is low (early stage)
    # Let's keep it simple: beta_ucb = 1.5 + 1.0 * progress
    # Early: beta ~ 2.5, Late: beta ~ 1.5
    beta_ucb = 1.5 + 1.0 * progress
    
    ucb_term = -mu + beta_ucb * sigma_safe
    
    # Combine EI and UCB
    # Weights: EI is usually more stable. UCB helps escape local minima.
    # Let's try equal weight or slightly favor EI.
    # score = (ei + ucb_term) / cost_penalty
    # However, EI and UCB have different scales.
    # EI is in objective units. UCB is in objective units.
    # So direct addition is reasonable.
    
    cost_penalty = np.sqrt(cost_safe)
    scores = (ei + ucb_term) / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
