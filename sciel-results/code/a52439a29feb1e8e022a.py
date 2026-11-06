"""Particle filter with kernel density estimation for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Convert to float arrays
    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    obs_var = float(observation_variance)
    
    # Step 1: Compute particle weights based on observation likelihood
    # For each particle, compute log-likelihood of observations
    X_obs = X[:, obs_idx]  # (n_members, n_obs)
    # Gaussian likelihood: log p(y|x) = -0.5 * sum((y - x_obs)^2 / obs_var)
    diff = y[None, :] - X_obs  # (n_members, n_obs)
    log_lik = -0.5 * np.sum(diff ** 2 / obs_var, axis=1)
    
    # Normalize weights using log-sum-exp for numerical stability
    max_log_lik = np.max(log_lik)
    log_weights = log_lik - max_log_lik
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Step 2: Importance resampling (stratified)
    # Use systematic resampling to reduce variance
    u = (np.arange(n_members) + np.random.uniform(0, 1)) / n_members
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # ensure it ends at 1
    indices = np.searchsorted(cdf, u)
    indices = np.clip(indices, 0, n_members - 1)
    
    # Resampled particles
    X_resampled = X[indices, :]
    
    # Step 3: Kernel density estimation to update all variables
    # Use the resampled particles to estimate the posterior
    # For observed variables, blend with observations using a kernel
    # For unobserved variables, use the resampled ensemble mean with spread
    
    # Compute the posterior mean for each variable
    posterior_mean = np.mean(X_resampled, axis=0)
    posterior_std = np.std(X_resampled, axis=0)
    
    # For observed indices, apply a soft update towards observations
    # The kernel bandwidth is related to the observation noise
    bandwidth = np.sqrt(obs_var) * 0.5
    
    # Build the updated ensemble
    updated = np.zeros_like(X_resampled)
    
    # For each particle, create a new state
    for i in range(n_members):
        updated[i, :] = X_resampled[i, :]
        # For observed variables, nudge towards observation with kernel weighting
        for j, idx in enumerate(obs_idx):
            # Gaussian kernel weight
            diff_val = y[j] - X_resampled[i, idx]
            kernel_weight = np.exp(-0.5 * (diff_val / bandwidth) ** 2)
            # Blend: move towards observation based on kernel weight
            updated[i, idx] = X_resampled[i, idx] + kernel_weight * (y[j] - X_resampled[i, idx])
    
    # Step 4: Ensure finite values and clip to [-100, 100]
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > DEADLINE:
        return updated
    
    # Step 5: Add a small perturbation to maintain ensemble spread
    # Use the posterior standard deviation as a guide
    spread_factor = 0.1
    for i in range(n_members):
        for j in range(n_vars):
            std_j = posterior_std[j]
            if std_j > 1e-6:
                updated[i, j] += spread_factor * std_j * np.random.randn()
    
    # Re-clip after perturbation
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated
# EVOLVE-BLOCK-END
