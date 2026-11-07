"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Probability of Being the Minimum (PBM).
    Assumes independence between candidate points for the probability calculation.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units).
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best (lowest) observed value so far. (Not used in PBM directly, but kept for interface)
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit
        return -mu.copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        mu = np.where(mask, mu, mu[mask].max() if np.any(mask) else 0.0)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate Probability of Being the Minimum (PBM)
    # P(y_i = min) approx prod_{j != i} P(y_i < y_j)
    # P(y_i < y_j) = Phi((mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2))
    
    # Create pairwise matrices
    # mu_diff[i, j] = mu[i] - mu[j]
    mu_diff = mu[:, np.newaxis] - mu[np.newaxis, :]
    
    # sigma_diff[i, j] = sqrt(sigma[i]^2 + sigma[j]^2)
    sigma_sq = sigma[:, np.newaxis] ** 2 + sigma[np.newaxis, :] ** 2
    sigma_diff = np.sqrt(sigma_sq)
    
    # Avoid division by zero
    sigma_diff[sigma_diff == 0] = 1e-12
    
    # Z scores
    z_scores = mu_diff / sigma_diff
    
    # Probabilities
    # Phi(z)
    probs = norm.cdf(z_scores)
    
    # Clip to avoid log(0)
    eps = 1e-10
    probs = np.clip(probs, eps, 1.0)
    
    # Log probabilities
    log_probs = np.log(probs)
    
    # Sum over j for each i (excluding i itself, where prob is 1, log is 0)
    # The diagonal is Phi(0) = 0.5, log(0.5) != 0.
    # We want prod_{j != i}. So we should exclude the diagonal.
    # Set diagonal log_probs to 0.
    np.fill_diagonal(log_probs, 0.0)
    
    log_scores = np.sum(log_probs, axis=1)
    
    # Convert back to probability space (or keep log if we just need ranking)
    # Since we return scores, and higher is better, we can return exp(log_scores)
    # Or just log_scores, as exp is monotonic.
    # However, the scale might matter if the evaluator does anything else?
    # "Return 2000 finite scores, higher meaning evaluate next".
    # The evaluator just picks the argmax. So relative order matters.
    # log_scores is sufficient.
    
    scores = log_scores
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
