"""Cost-aware acquisition function for Bayesian optimization on 3-hump camel 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = more desirable)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / np.sqrt(cost_safe)
    
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Core: Cost-Adjusted Expected Improvement (EI / sqrt(cost)) ---
    # This is the measured improvement strategy from the lessons.
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by sqrt(cost) to balance improvement vs. expense
    # Using sqrt(cost) instead of cost gives a softer penalty, allowing
    # some exploration in moderately expensive regions when EI is high.
    scores = ei / np.sqrt(cost)
    
    # --- Structural addition: Thompson Sampling-style stochastic exploration ---
    # Instead of a deterministic UCB term, we inject controlled randomness
    # proportional to uncertainty. This is a different search paradigm:
    # rather than always picking the argmax of a deterministic score,
    # we add a small Gaussian perturbation scaled by sigma to occasionally
    # sample from high-uncertainty regions. This helps escape local optima
    # without the structured bias of UCB.
    #
    # We use a small fixed seed per call to make the randomness reproducible
    # within a single acquisition call (2000 candidates get consistent noise),
    # but different across calls to provide stochastic diversity.
    #
    # The noise is scaled to be small relative to the EI signal so that
    # exploitation dominates, but large enough to occasionally shift the
    # ranking in high-uncertainty regions.
    
    # Determine a noise scale that is small relative to typical EI values
    ei_median = np.median(ei[ei > 0]) if np.any(ei > 0) else 1.0
    noise_scale = 0.05 * ei_median  # 5% of median EI
    
    # Generate deterministic pseudo-random noise using a hash of the input
    # This avoids needing a random state while still providing diversity
    # across different calls (mu, sigma change each call)
    # Use a simple approach: derive a seed from the data
    seed_val = int(np.sum(np.abs(mu[:10])) * 1000 + np.sum(np.abs(sigma[:10])) * 100) % (2**31)
    rng = np.random.RandomState(seed_val)
    noise = rng.normal(0, 1, size=mu.shape) * noise_scale * sigma / (np.max(sigma) + 1e-9)
    
    scores += noise
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
