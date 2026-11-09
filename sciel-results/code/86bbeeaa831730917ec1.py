"""Lorenz-96 ensemble assimilation with refined Stochastic Ensemble Kalman Filter (SE-Kalman)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Refined Stochastic Ensemble Kalman Filter (SE-Kalman) update for Lorenz-96.
    
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
    
    # Convert to float64 for precision
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    R_val = float(observation_variance)
    
    # Step 1: Localize observations using a Gaussian correlation function
    # Tighter correlation length (2.2) to reduce spurious correlations in chaotic system
    corr_length = 2.2
    
    # Compute localization weights for each observed variable to each state variable
    # Weights shape: (n_obs, n_vars)
    loc_weights = np.zeros((n_obs, n_vars))
    for j, obs_i in enumerate(obs_idx):
        for i in range(n_vars):
            # Periodic distance on the ring
            dist = abs(i - obs_i)
            dist = min(dist, n_vars - dist)
            loc_weights[j, i] = np.exp(-0.5 * (dist / corr_length) ** 2)
    
    # Step 2: Adaptive Inflation
    # Compute ensemble mean and variance at observed locations
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # Ensemble variance at observed locations
    H_ens = ens[:, obs_idx]
    H_mean = np.mean(H_ens, axis=0)
    H_anom = H_ens - H_mean
    ens_obs_var = np.sum(H_anom**2, axis=0) / (n_members - 1.0)
    
    # Adaptive inflation: target ensemble variance to be slightly larger than observation noise
    # to account for model error and ensure sufficient spread
    target_ratio = 1.2
    # Calculate inflation factor based on the ratio of current spread to target spread
    # If spread is too low, inflate more; if too high, deflate slightly or leave alone
    current_spread_ratio = np.mean(ens_obs_var) / R_val
    
    if current_spread_ratio < 0.5:
        inflation_factor = 1.05
    elif current_spread_ratio < 0.8:
        inflation_factor = 1.02
    elif current_spread_ratio > 1.5:
        inflation_factor = 0.98  # Deflate slightly if spread is too large
    else:
        inflation_factor = 1.0
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_centered
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # Step 3: Stochastic Ensemble Kalman Filter Update
    # Precompute the ensemble size factor
    alpha = n_members / (n_members - 1.0)
    
    # We process each observation sequentially (sequential assimilation)
    # This is often more stable than simultaneous assimilation for small ensembles
    for j in range(n_obs):
        obs_j = obs[j]
        obs_idx_j = obs_idx[j]
        
        # Localization weights for this observation
        w = loc_weights[j, :]  # shape (n_vars,)
        
        # If the weight for the observed variable itself is too small, skip (should not happen for self)
        if w[obs_idx_j] < 0.01:
            continue
        
        # Current ensemble values for this observation
        H_ens_j = ens[:, obs_idx_j]  # shape (n_members,)
        H_mean_j = np.mean(H_ens_j)
        H_anom_j = H_ens_j - H_mean_j
        
        # Compute S (observation innovation variance)
        # S = P_oo + R
        P_oo = alpha * np.dot(H_anom_j, H_anom_j) / (n_members - 1.0)
        S = P_oo + R_val
        
        if S <= 0:
            continue
            
        # Generate a random rotation matrix for stochastic perturbation
        # This ensures that the ensemble mean is preserved in expectation
        # and the covariance is updated correctly
        Q, _ = np.linalg.qr(np.random.randn(n_members, n_members))
        
        # Create perturbed observations: y_pert = y + Q @ noise
        # noise ~ N(0, R_val)
        noise_vector = np.random.randn(n_members) * np.sqrt(R_val)
        perturbed_obs = obs_j + Q @ noise_vector
        
        # Update the ensemble
        # For each state variable i:
        # P_xo = Cov(x_i, H(x_j))
        # K = P_xo / S
        # x_new = x + K * (y_pert - H(x))
        
        # Vectorized calculation for all variables at once
        # x_anom shape: (n_members, n_vars)
        x_anom = ens - ens_mean  # This is slightly incorrect as ens_mean changes per step, 
                                 # but we update ens in place. 
                                 # It's better to recompute anomalies for each variable or 
                                 # accept that sequential updates use the current ensemble state.
        
        # In sequential EnKF, we typically update one variable at a time or use a 
        # vectorized approach that accounts for the sequential nature. 
        # Here, we compute cross-covariances with the *current* ensemble state.
        
        # Cross-covariance matrix: (n_vars, 1) effectively, but we can vectorize over variables
        # P_xo[i] = Cov(x_i, H(x_j))
        # We need to compute this for all i where w[i] > threshold
        
        # Let's stick to a loop over variables for clarity and correctness in sequential update,
        # as vectorizing across variables when the ensemble is being updated sequentially 
        # requires careful handling of which anomalies are used.
        
        for i in range(n_vars):
            if w[i] < 0.01:
                continue
            
            # Compute cross-covariance between variable i and the observation
            # Use the current ensemble state for variable i
            x_i = ens[:, i]
            x_i_mean = np.mean(x_i)
            x_i_anom = x_i - x_i_mean
            
            cross_cov = alpha * np.dot(x_i_anom, H_anom_j) / (n_members - 1.0)
            
            # Apply localization
            cross_cov_loc = cross_cov * w[i]
            
            # Stochastic update
            gain = cross_cov_loc / S
            ens[:, i] += gain * (perturbed_obs - H_ens_j)
            
            # Update H_ens_j if i == obs_idx_j? 
            # No, H_ens_j is the observation operator applied to the ensemble.
            # In sequential assimilation, if we update the observed variable itself,
            # H_ens_j should ideally be updated for the next variable's calculation?
            # Actually, in the standard sequential SE-Kalman, we update the state vector.
            # The observation operator H is linear here (identity for observed components).
            # The innovation (y_pert - H(x)) uses the *current* x.
            # Since we are updating x in place, subsequent variables will use the updated
            # mean/spread of the previous variables, which is correct for sequential assimilation.
            # However, H_ens_j (the column of the ensemble for the observed variable) 
            # does not change if i != obs_idx_j. 
            # If i == obs_idx_j, H_ens_j would change, but we only use H_ens_j to compute
            # the innovation for the *current* update step for variable i.
            # Wait, the innovation is (y_pert - H(x)). H(x) is x[obs_idx_j].
            # If i == obs_idx_j, we are updating x[obs_idx_j]. The innovation uses the
            # *pre-update* value of x[obs_idx_j] for the calculation of the gain and the
            # innovation term itself in the standard formulation?
            # Actually, the standard update is:
            # x_new = x + K * (y_pert - H(x))
            # This uses the old x for H(x).
            # So H_ens_j should be the values *before* the update for this specific observation.
            # We stored H_ens_j at the start of the j-loop, so it is fixed for all i in this loop.
            # This is correct for the innovation term.
            # The gain K is computed from the pre-update covariance, which is also correct.
            
        # Update ensemble mean for the next iteration?
        # The mean changes slightly due to the stochastic nature, but in expectation it is preserved.
        # We don't strictly need to recompute ens_mean for the next observation's H_ens_j
        # because H_ens_j is taken directly from ens[:, obs_idx_j] at the start of the j-loop.
        # However, the inflation and cross-covariances rely on the current state.
        
    # Step 4: Final Inflation and Clipping
    # Apply a small multiplicative inflation to compensate for any underestimation of spread
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    ens = ens_mean + 1.01 * ens_centered
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finite values
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    # Check time
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
