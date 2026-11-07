"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function via Monte Carlo approximation.
    
    KG estimates the expected reduction in the global minimum value
    by considering the joint distribution of the 2000 candidates.
    We use a Monte Carlo approximation with a moderate number of samples
    to estimate the KG efficiently.
    
    For each candidate x_i, KG(x_i) = E[min_j y_j - min_j y_j']
    where y_j is the posterior at candidate j, and y_j' is the posterior
    at candidate j after observing x_i.
    
    We approximate this by:
    1. Sampling S times from the posterior at a subset of K candidates.
    2. For each sample, compute the minimum value.
    3. For each candidate i, compute the expected minimum given that we
       observe x_i, using the fact that the posterior at x_i is known.
    
    Parameters:
    mu (np.array): Posterior mean at candidate points, shape (N,)
    sigma (np.array): Posterior standard deviation at candidate points, shape (N,)
    f_best (float): Current best observed value
    
    Returns:
    np.array: Scores for each candidate, shape (N,). Higher is better.
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Number of Monte Carlo samples
    # Use a moderate number to balance accuracy and speed
    n_samples = 200
    
    # We'll sample from all n candidates for simplicity, but use a subset
    # if n is very large. Here n=2000, so we sample from all.
    
    # Precompute z-scores for efficiency
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement as a baseline
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # KG approximation via Monte Carlo
    # For each sample s, draw y_j ~ N(mu_j, sigma_j^2) for all j
    # min_s = min_j y_j
    # For candidate i, the KG is E[min(min_s, y_i) - min_s]
    # where y_i is drawn from N(mu_i, sigma_i^2)
    
    # Vectorized MC: shape (n_samples, n)
    # Draw samples
    samples = mu[None, :] + sigma[None, :] * np.random.randn(n_samples, n)
    
    # min across candidates for each sample
    min_samples = np.min(samples, axis=1)  # shape (n_samples,)
    
    # For each candidate i, compute E[min(min_samples, y_i) - min_samples]
    # where y_i ~ N(mu_i, sigma_i^2)
    # This is E[min(M, Y) - M] where M = min_samples, Y ~ N(mu_i, sigma_i^2)
    # = E[min(M, Y)] - E[M]
    # E[min(M, Y)] = E[M * 1(M < Y) + Y * 1(Y < M)]
    # = E[M * Phi((M - mu_i)/sigma_i) + sigma_i * phi((M - mu_i)/sigma_i) + mu_i * (1 - Phi((M - mu_i)/sigma_i))]
    
    # For each sample s and candidate i:
    # z_si = (min_s - mu_i) / sigma_i
    # E[min(M, Y) | M] = M * Phi(z_si) + sigma_i * phi(z_si) + mu_i * (1 - Phi(z_si))
    
    # Vectorized computation
    # min_samples shape: (n_samples,)
    # mu, sigma shape: (n,)
    # z_si shape: (n_samples, n)
    
    z_si = (min_samples[:, None] - mu[None, :]) / sigma[None, :]
    z_si = np.clip(z_si, -10.0, 10.0)
    
    # Phi and phi
    Phi_si = norm.cdf(z_si)  # shape (n_samples, n)
    phi_si = norm.pdf(z_si)  # shape (n_samples, n)
    
    # E[min(M, Y) | M] for each sample and candidate
    # = M * Phi + sigma * phi + mu * (1 - Phi)
    e_min_given_m = min_samples[:, None] * Phi_si + sigma[None, :] * phi_si + mu[None, :] * (1.0 - Phi_si)
    
    # E[M] is just the mean of min_samples
    e_m = np.mean(min_samples)
    
    # KG for each candidate i = E[min(M, Y)] - E[M]
    # = mean over samples of e_min_given_m[:, i] - e_m
    kg = np.mean(e_min_given_m, axis=0) - e_m
    
    # Ensure kg is non-negative (it should be, but numerical issues may cause negatives)
    kg = np.maximum(kg, 0.0)
    
    # Combine KG with EI for stability
    # KG can be noisy, so we blend with EI
    score = kg + 0.5 * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
