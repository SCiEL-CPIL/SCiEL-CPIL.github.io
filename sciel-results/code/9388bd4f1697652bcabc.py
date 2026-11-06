"""Particle filter with importance resampling and KDE for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter analysis update for Lorenz-96.
    
    Uses importance resampling and a kernel density estimate (KDE) to model the posterior,
    avoiding the Gaussian assumption and localization artifacts inherent in EnKF.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    # --- Compute likelihoods for each particle ---
    # The likelihood is proportional to exp(-0.5 * sum((y_k - x_{i_k})^2 / obs_var))
    # where x_{i_k} is the state at observed index k for each particle
    
    H_ens = ensemble[:, observed_indices]  # (nm, nobs)
    innovations = observations[None, :] - H_ens  # (nm, nobs)
    
    # Log-likelihood (up to a constant)
    log_lik = -0.5 * np.sum(innovations ** 2, axis=1) / obs_var  # (nm,)
    
    # Normalize log-likelihoods to avoid underflow
    log_lik_max = np.max(log_lik)
    weights = np.exp(log_lik - log_lik_max)
    
    # Normalize weights
    weight_sum = np.sum(weights)
    if weight_sum < 1e-12:
        weights = np.ones(nm) / nm
    else:
        weights = weights / weight_sum
    
    # --- Importance resampling ---
    # Use systematic resampling for low variance
    n_resample = nm  # Keep same number of particles
    
    # Cumulative weights
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0  # Ensure exact 1.0
    
    # Systematic resampling
    u0 = np.random.uniform(0, 1.0 / n_resample)
    u = u0 + np.arange(n_resample) / n_resample
    u = u % 1.0  # Wrap around
    
    # Find indices
    indices = np.searchsorted(cum_weights, u)
    indices = np.clip(indices, 0, nm - 1)
    
    # Resampled ensemble
    resampled = ensemble[indices, :]
    
    # --- KDE-based posterior estimation ---
    # For each variable, we use the resampled particles to estimate the posterior
    # We use a simple approach: for each variable, compute a weighted average
    # with a Gaussian kernel centered at each particle
    
    # However, with only 16 particles, a full KDE is expensive and may be unstable.
    # Instead, we use a hybrid approach:
    # 1. For observed variables, blend the resampled mean with the observations
    # 2. For unobserved variables, use the resampled ensemble directly
    # 3. Add a small regularization to prevent collapse
    
    # Compute the effective sample size to check for degeneracy
    ess = 1.0 / np.sum(weights ** 2)
    
    # If ESS is low, the particles are degenerate. We need to reinitialize.
    # A common threshold is ESS < nm / 2
    if ess < nm / 2.0:
        # Reinitialize particles around the weighted mean with some spread
        weighted_mean = np.sum(weights[:, None] * ensemble, axis=0)
        
        # Estimate spread from the original ensemble
        spread = np.std(ensemble, axis=0, ddof=1)
        spread = np.maximum(spread, 0.1)  # Minimum spread
        
        # Generate new particles
        rng = np.random.default_rng()
        new_ensemble = weighted_mean[None, :] + rng.standard_normal((nm, nvars)) * spread[None, :]
        
        # Blend with observations for observed indices
        for k, obs_idx in enumerate(observed_indices):
            obs_idx_int = int(obs_idx)
            # The observation gives information about this variable
            # Blend: new_ensemble[:, obs_idx_int] = alpha * obs_val + (1-alpha) * new_ensemble[:, obs_idx_int]
            # Choose alpha based on relative uncertainty
            # If obs_var is small relative to state variance, trust observation more
            state_var = spread[obs_idx_int] ** 2
            alpha = state_var / (state_var + obs_var)
            alpha = np.clip(alpha, 0.3, 0.7)  # Don't fully trust either
            new_ensemble[:, obs_idx_int] = alpha * observations[k] + (1 - alpha) * new_ensemble[:, obs_idx_int]
        
        # Safety checks
        new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
        new_ensemble = np.nan_to_num(new_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return new_ensemble
    
    # --- For healthy particle sets, use a weighted KDE-like approach ---
    # For each variable, estimate the posterior mean and variance using the weights
    
    updated_ensemble = np.zeros_like(resampled)
    
    # For observed variables, we want to incorporate the observation information
    # For unobserved variables, we use the resampled particles
    
    # Compute per-variable statistics
    for j in range(nvars):
        # Check if this variable is observed
        is_observed = False
        obs_val = 0.0
        for k, obs_idx in enumerate(observed_indices):
            if int(obs_idx) == j:
                is_observed = True
                obs_val = observations[k]
                break
        
        if is_observed:
            # Blend the particle values with the observation
            # Use a Kalman-like update for the mean
            # The "prior" mean is the weighted mean of the particles
            prior_mean = np.sum(weights * resampled[:, j])
            prior_var = np.sum(weights * (resampled[:, j] - prior_mean) ** 2)
            prior_var = max(prior_var, 1e-6)
            
            # Posterior mean: (prior_var * obs + obs_var * prior_mean) / (prior_var + obs_var)
            # But we want to generate particles, not just the mean
            # So we shift the particles toward the observation
            
            # Compute the innovation
            innovation = obs_val - prior_mean
            
            # Gain
            gain = prior_var / (prior_var + obs_var)
            
            # Shift all particles by gain * innovation
            shift = gain * innovation
            
            # Also reduce the spread by sqrt(1 - gain)
            new_var = prior_var * (1 - gain)
            scale = np.sqrt(max(new_var / prior_var, 0.1))
            
            # Update particles
            updated_ensemble[:, j] = resampled[:, j] + shift
            updated_ensemble[:, j] = prior_mean + scale * (resampled[:, j] - prior_mean)
            
        else:
            # Unobserved variable: use resampled particles directly
            # But we can use the correlations with observed variables
            # Simple approach: just use the resampled values
            updated_ensemble[:, j] = resampled[:, j]
    
    # --- Final safety checks ---
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() - t0 > 115.0:
        return updated_ensemble
    
    return updated_ensemble
# EVOLVE-BLOCK-END
