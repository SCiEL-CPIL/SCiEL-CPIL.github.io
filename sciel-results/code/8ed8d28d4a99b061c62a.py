import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter with overlapping blocks and adaptive inflation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    x = ensemble.copy()
    
    # --- Adaptive Inflation Estimation ---
    # We estimate a global inflation factor based on the total innovation variance
    x_mean = np.mean(x, axis=0)
    anomalies = x - x_mean
    S = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Global innovation variance (before inflation)
    # S_HH is the model covariance of observations
    S_HH = S[np.ix_(observed_indices, observed_indices)]
    # Theoretical innovation variance
    var_inn_model = np.trace(S_HH) / n_obs + observation_variance
    # Actual innovation variance
    innovation = observations - x_mean[observed_indices]
    var_inn_obs = np.var(innovation, ddof=1) if n_obs > 1 else np.var(innovation)
    
    # Target inflation factor
    # We want var_inn_obs to be close to var_inn_model * inflation^2
    # So inflation = sqrt(var_inn_obs / var_inn_model)
    # Clip to prevent explosion
    if var_inn_model > 0:
        inflation_factor = np.sqrt(var_inn_obs / var_inn_model)
    else:
        inflation_factor = 1.0
    
    inflation_factor = np.clip(inflation_factor, 1.0, 2.0)
    
    # Apply inflation to anomalies
    x_inflated = x_mean + anomalies * inflation_factor
    
    # --- Localized EnKF with Overlapping Blocks ---
    # Define blocks
    block_size = 10
    stride = 5
    n_blocks = n_vars // stride  # 4 blocks for n_vars=20
    
    # Storage for updated values and counts
    updated_values = np.zeros((n_members, n_vars))
    counts = np.zeros(n_vars)
    
    for b in range(n_blocks):
        # Start index for this block
        start = b * stride
        # State indices for this block (cyclic)
        state_indices = np.array([(start + i) % n_vars for i in range(block_size)])
        
        # Observation indices within this block
        # We only use observations that are in the state_indices of this block
        obs_in_block_mask = np.zeros(n_obs, dtype=bool)
        for j_idx, obs_idx in enumerate(observed_indices):
            if obs_idx in state_indices:
                obs_in_block_mask[j_idx] = True
        
        if not np.any(obs_in_block_mask):
            # No observations in this block, skip (or just copy)
            # This shouldn't happen with even indices and size 10 blocks
            continue
            
        local_obs_indices = observed_indices[obs_in_block_mask]
        local_obs = observations[obs_in_block_mask]
        n_local_obs = len(local_obs)
        
        if n_local_obs == 0:
            continue
            
        # Extract local ensemble
        x_local = x_inflated[:, state_indices]
        
        # Compute local mean and anomalies
        x_local_mean = np.mean(x_local, axis=0)
        x_local_anom = x_local - x_local_mean
        
        # Local covariance
        S_local = (x_local_anom.T @ x_local_anom) / (n_members - 1)
        
        # Observation operator for local block
        # H maps state_indices to local_obs_indices
        # We need S_local_HH and S_local_xH
        # local_obs_indices are a subset of state_indices
        # Find positions of local_obs_indices within state_indices
        obs_pos_in_state = []
        for obs_idx in local_obs_indices:
            pos = np.where(state_indices == obs_idx)[0]
            if len(pos) > 0:
                obs_pos_in_state.append(pos[0])
        
        if len(obs_pos_in_state) == 0:
            continue
            
        obs_pos_in_state = np.array(obs_pos_in_state)
        
        S_HH_local = S_local[np.ix_(obs_pos_in_state, obs_pos_in_state)]
        S_xH_local = S_local[:, obs_pos_in_state]
        
        # Add observation noise
        S_HH_local += observation_variance * np.eye(n_local_obs)
        
        # Compute Gain
        try:
            S_HH_inv = np.linalg.inv(S_HH_local)
            K_local = S_xH_local @ S_HH_inv
        except np.linalg.LinAlgError:
            S_HH_inv = np.linalg.pinv(S_HH_local)
            K_local = S_xH_local @ S_HH_inv
            
        # Innovation
        innovation_local = local_obs - x_local_mean[obs_pos_in_state]
        
        # Update mean
        x_local_mean_new = x_local_mean + K_local @ innovation_local
        
        # Update ensemble members
        x_local_new = np.zeros_like(x_local)
        for k in range(n_members):
            anom_k = x_local_anom[k]
            H_anom = anom_k[obs_pos_in_state]
            delta = K_local @ H_anom
            x_local_new[k] = x_local_mean_new + anom_k - delta
            
        # Accumulate updates
        # For each variable in this block, add the updated value to updated_values
        # and increment count
        for i, idx in enumerate(state_indices):
            updated_values[:, idx] += x_local_new[:, i]
            counts[idx] += 1
            
    # Average the updates for variables that were in multiple blocks
    # Variables not in any block (shouldn't happen) keep original value
    for i in range(n_vars):
        if counts[i] > 0:
            x[:, i] = updated_values[:, i] / counts[i]
        else:
            # Fallback to inflated value
            x[:, i] = x_inflated[:, i]
            
    # Final safety checks
    x = np.clip(x, -100.0, 100.0)
    x = np.where(np.isfinite(x), x, 0.0)
    
    return x
