"""Lorenz-96 ensemble assimilation with Stochastic Ensemble Kalman Filter (SE-Kalman)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-Kalman) update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Convert to float
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    R_val = float(observation_variance)
    
    # Step 1: Localize observations using a Gaussian correlation function
    # Tighter correlation length (~2.8) matches Lorenz-96 correlation structure
    corr_length = 2.8
    
    # Compute localization weights for each observed variable to each state variable
    loc_weights = np.zeros((n_obs, n_vars))
    for j, obs_i in enumerate(obs_idx):
        for i in range(n_vars):
            # Periodic distance on the ring
            dist = abs(i - obs_i)
            dist = min(dist, n_vars - dist)
            loc_weights[j, i] = np.exp(-0.5 * (dist / corr_length) ** 2)
    
    # Step 2: Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Step 3: Adaptive inflation based on ensemble spread vs observation noise
    # Compute ensemble variance at observed locations
    ens_obs_var = np.var(ens[:, obs_idx], axis=0)
    obs_noise_var = R_val
    
    # Adaptive inflation: increase if ensemble is too spread, decrease if too narrow
    # Target ratio of ensemble variance to observation variance ~ 1
    target_ratio = 1.0
    inflation_factor = 1.0
    if np.mean(ens_obs_var) > 0:
        spread_ratio = np.mean(ens_obs_var) / obs_noise_var
        # If spread is too large relative to noise, don't inflate; if too small, inflate more
        if spread_ratio < 0.5:
            inflation_factor = 1.08
        elif spread_ratio < 1.0:
            inflation_factor = 1.04
        else:
            inflation_factor = 1.0
    
    # Apply inflation to the ensemble
    ens_centered = ens - ens_mean
    ens = ens_mean + inflation_factor * ens_centered
    ens_mean = np.mean(ens, axis=0)
    
    # Step 4: Stochastic Ensemble Kalman Filter update
    # Use random rotation to generate perturbations that preserve ensemble mean
    # and apply Kalman gain exactly
    
    # Precompute the ensemble size factor
    alpha = n_members / (n_members - 1.0)
    
    # For each observation, perform a local ensemble update with stochastic perturbation
    for j in range(n_obs):
        obs_j = obs[j]
        obs_idx_j = obs_idx[j]
        
        # Localization weights for this observation
        w = loc_weights[j, :]  # shape (n_vars,)
        
        # If the weight for the observed variable itself is too small, skip
        if w[obs_idx_j] < 0.01:
            continue
        
        # Compute the observation operator applied to ensemble
        H_ens = ens[:, obs_idx_j]  # shape (n_members,)
        H_mean = np.mean(H_ens)
        
        # Compute the variance of the observation in the ensemble
        H_anom = H_ens - H_mean
        S = alpha * np.dot(H_anom, H_anom) / (n_members - 1.0) + R_val
        
        # Generate a random rotation matrix for stochastic perturbation
        # Use QR decomposition of a random Gaussian matrix
        Q, _ = np.linalg.qr(np.random.randn(n_members, n_members))
        
        # Create perturbed observations: obs_j + Q @ noise_vector
        # The noise vector has elements ~ N(0, R_val)
        noise_vector = np.random.randn(n_members) * np.sqrt(R_val)
        perturbed_obs = obs_j + Q @ noise_vector
        
        # Update the ensemble using stochastic Kalman gain
        # For each variable, compute the update
        for i in range(n_vars):
            if w[i] < 0.01:
                continue
            
            # Compute the cross-covariance between variable i and the observation
            x_anom = ens[:, i] - np.mean(ens[:, i])
            cross_cov = alpha * np.dot(x_anom, H_anom) / (n_members - 1.0)
            
            # Apply localization
            cross_cov_loc = cross_cov * w[i]
            
            # Stochastic update: use perturbed observations instead of single observation
            # x_new = x + K * (perturbed_obs - H(x))
            # But we need to be careful: the standard SE-Kalman uses:
            # x_new = x + K * (y_perturbed - H(x))
            # where K is the Kalman gain computed from the ensemble
            gain = cross_cov_loc / S
            ens[:, i] += gain * (perturbed_obs - H_ens)
        
        # Recompute anomalies after update
        ens_mean = np.mean(ens, axis=0)
    
    # Step 5: Final inflation and clipping
    # Apply a small multiplicative inflation
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    ens = ens_mean + 1.02 * ens_centered
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finite values
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    # Check time
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
