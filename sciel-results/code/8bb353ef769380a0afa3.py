"""Particle filter with resampling and KDE for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter analysis update for Lorenz-96.
    
    Uses importance resampling and a kernel density estimate (KDE) to model the posterior,
    avoiding the Gaussian assumption inherent in EnKF while handling non-Gaussian tails.
    
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
    nobs = len(observed_indices)
    obs_var = float(observation_variance)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    # --- Step 1: Compute importance weights based on observation likelihood ---
    ens_mean = np.mean(ensemble, axis=0)
    ens_anoms = ensemble - ens_mean
    
    # Compute per-member log-likelihood under Gaussian observation model
    # For each member, compute innovation: obs - H*x
    # H is identity on observed_indices
    H_ens = ensemble[:, observed_indices]  # (nm, nobs)
    innovations = observations[None, :] - H_ens  # (nm, nobs)
    
    # Log likelihood: -0.5 * sum((y - Hx)^2 / obs_var) - 0.5 * nobs * log(2*pi*obs_var)
    log_lik = -0.5 * np.sum(innovations ** 2, axis=1) / obs_var
    # Subtract mean for numerical stability
    log_lik -= np.max(log_lik)
    weights = np.exp(log_lik)
    
    # Normalize weights
    weight_sum = np.sum(weights)
    if weight_sum < 1e-10:
        weights = np.ones(nm) / nm
    else:
        weights /= weight_sum
    
    # --- Step 2: Effective sample size check and resampling ---
    eff_sample_size = 1.0 / np.sum(weights ** 2)
    
    # If effective sample size is too low, resample
    resample_threshold = 0.5 * nm
    if eff_sample_size < resample_threshold:
        # Systematic resampling
        r = np.random.uniform(0, 1.0 / nm)
        u = (r + np.arange(nm)) / nm
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0  # ensure exact 1.0
        indices = np.searchsorted(cdf, u)
        indices = np.clip(indices, 0, nm - 1)
        ensemble = ensemble[indices]
        weights = np.ones(nm) / nm
    
    # --- Step 3: Adaptive inflation based on spread ---
    # Estimate if ensemble is under-dispersive
    ens_obs_var = np.var(ensemble[:, observed_indices], axis=0, ddof=1)
    mean_ens_var = np.mean(ens_obs_var)
    obs_innov_var = np.var(observations - ens_mean[observed_indices], ddof=1)
    
    if mean_ens_var > 1e-6:
        target_alpha = (obs_innov_var - obs_var) / mean_ens_var
        target_alpha = np.clip(target_alpha, 1.0, 2.5)
    else:
        target_alpha = 2.0
    
    inflation_factor = np.sqrt(target_alpha)
    ens_mean_after_infl = np.mean(ensemble, axis=0)
    ensemble = ens_mean_after_infl + inflation_factor * (ensemble - ens_mean_after_infl)
    
    # --- Step 4: KDE-based posterior sampling ---
    # Use a Gaussian KDE with adaptive bandwidth based on ensemble spread
    # For each dimension, estimate bandwidth from the ensemble
    ens_std = np.std(ensemble, axis=0, ddof=1)
    # Adaptive bandwidth: scale by a factor to capture non-Gaussian structure
    bandwidth = np.maximum(ens_std * 0.5, 0.1)
    
    # Generate new particles using the KDE
    # For each new particle, pick a source particle with probability proportional to weight,
    # then add Gaussian noise with the local bandwidth
    rng = np.random.RandomState(42)
    
    # Resample indices based on weights
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    u = rng.uniform(0, 1.0, size=nm)
    source_indices = np.searchsorted(cdf, u)
    source_indices = np.clip(source_indices, 0, nm - 1)
    
    # Create new ensemble by perturbing source particles
    updated_ensemble = np.zeros_like(ensemble)
    for m in range(nm):
        src = source_indices[m]
        # Add noise with dimension-dependent bandwidth
        noise = rng.normal(0, 1, size=nvars) * bandwidth
        updated_ensemble[m] = ensemble[src] + noise
    
    # --- Step 5: Second pass - refine with observation constraint ---
    # For each observation, nudge particles toward consistency
    for k, obs_idx in enumerate(observed_indices):
        obs_val = observations[k]
        obs_idx_int = int(obs_idx)
        
        # Local window
        local_radius = 3
        local_indices = []
        for d in range(-local_radius, local_radius + 1):
            local_indices.append((obs_idx_int + d) % nvars)
        local_indices = np.array(list(set(local_indices)))
        
        local_ens = updated_ensemble[:, local_indices]
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        # Find position of obs_idx in local_indices
        try:
            center_pos = np.where(local_indices == obs_idx_int)[0][0]
        except IndexError:
            continue
        
        # Compute local Kalman gain for refinement
        n_local = len(local_indices)
        C_local = (local_anoms.T @ local_anoms) / (nm - 1)
        H_local = np.zeros(n_local)
        H_local[center_pos] = 1.0
        P_oo = H_local @ C_local @ H_local + obs_var
        K_local = (C_local @ H_local) / P_oo
        
        # Apply soft update: blend between particle and Kalman-corrected value
        # Use a mixing factor to preserve particle diversity
        mix_factor = 0.3
        for m in range(nm):
            H_x_f = local_mean[center_pos] + local_anoms[m, center_pos]
            innov_m = obs_val - H_x_f
            kalman_update = K_local * innov_m
            # Blend: keep particle structure but nudge toward observation
            local_ens[m] = (1 - mix_factor) * local_ens[m] + mix_factor * (local_ens[m] + kalman_update)
        
        updated_ensemble[:, local_indices] = local_ens
        
        # Deadline check
        if time.time() - t0 > 115.0:
            break
    
    # --- Step 6: Final safety checks ---
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
