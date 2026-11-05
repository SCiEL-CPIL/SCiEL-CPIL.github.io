"""Lorenz-96 assimilation using sequential coordinate-wise EnSRF regression."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Square Root Filter analysis update using
    sequential coordinate-wise regression.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Make a copy to work with
    result = ensemble.copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(result, axis=0)
    
    # Observation error standard deviation
    obs_std = np.sqrt(observation_variance)
    
    # Process each observed coordinate sequentially using regression
    # For each observed index, regress the unobserved variables onto it
    # using the ensemble covariance, with localization
    
    # Localization parameters
    decay_length = 3.0  # Gaussian decay length scale
    
    for obs_j in range(n_obs):
        if time.time() > deadline:
            break
            
        obs_idx = observed_indices[obs_j]
        obs_val = observations[obs_j]
        
        # Compute the current mean of the observed variable
        current_mean_obs = np.mean(result[:, obs_idx])
        
        # Compute anomalies
        anomalies = result - np.mean(result, axis=0)
        
        # Compute the sample standard deviation of the observed variable
        var_obs = np.var(anomalies[:, obs_idx], ddof=1)
        
        if var_obs < 1e-12:
            continue
        
        std_obs = np.sqrt(var_obs)
        
        # For each unobserved variable, compute the regression coefficient
        # and update the ensemble
        for i in range(n_state):
            if i == obs_idx:
                continue
            
            # Circular distance for localization
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            
            # Gaussian localization weight
            w = np.exp(-d**2 / (2.0 * decay_length**2))
            
            if w < 0.01:
                continue
            
            # Compute the correlation between variable i and the observed variable
            cov_ij = np.mean(anomalies[:, i] * anomalies[:, obs_idx], ddof=1)
            std_i = np.sqrt(np.var(anomalies[:, i], ddof=1))
            
            if std_i < 1e-12:
                continue
            
            # Correlation coefficient
            corr = cov_ij / (std_i * std_obs)
            
            # Regression coefficient (with localization)
            # The update is: x_i_new = x_i + beta * (obs - x_obs)
            # where beta = w * corr * std_i / std_obs
            beta = w * corr * std_i / std_obs
            
            # Update each ensemble member
            for m in range(n_members):
                result[m, i] += beta * (obs_val - result[m, obs_idx])
        
        # Now update the observed variable itself to match the observation
        # This is done by shifting the ensemble so that the mean matches
        # the observation, but we also need to account for the observation error
        
        # The analysis variance for the observed variable:
        # 1/sigma_a^2 = 1/sigma_f^2 + 1/sigma_obs^2
        # sigma_a^2 = (sigma_f^2 * sigma_obs^2) / (sigma_f^2 + sigma_obs^2)
        
        var_f = np.var(result[:, obs_idx], ddof=1)
        if var_f < 1e-12:
            continue
        
        # Analysis variance
        var_a = (var_f * observation_variance) / (var_f + observation_variance)
        std_a = np.sqrt(var_a)
        
        # The optimal mean is:
        # mean_a = (var_obs_weight * obs + var_f_weight * mean_f) / (var_f + var_obs)
        # where var_obs_weight = 1/var_f, var_f_weight = 1/var_obs
        mean_f = np.mean(result[:, obs_idx])
        
        # Weighted mean
        mean_a = (observation_variance * mean_f + var_f * obs_val) / (var_f + observation_variance)
        
        # Update the ensemble: shift to new mean and scale to new variance
        # x_new = mean_a + (x_old - mean_f) * (std_a / std_f)
        std_f = np.sqrt(var_f)
        
        if std_f > 1e-12:
            scale = std_a / std_f
            for m in range(n_members):
                result[m, obs_idx] = mean_a + (result[m, obs_idx] - mean_f) * scale
    
    # Adaptive inflation to prevent filter collapse
    # Compute innovation statistics
    ens_mean_final = np.mean(result, axis=0)
    
    # Estimate expected innovation variance from ensemble spread
    # For observed variables
    innov_vars = []
    expected_vars = []
    
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        innov = observations[j] - ens_mean_final[obs_idx]
        innov_vars.append(innov**2)
        
        # Expected innovation variance from ensemble
        ens_var = np.var(result[:, obs_idx], ddof=1)
        expected_vars.append(ens_var + observation_variance)
    
    innov_vars = np.array(innov_vars)
    expected_vars = np.array(expected_vars)
    
    # Inflation factor based on ratio of expected to actual
    valid = expected_vars > 1e-12
    if np.any(valid):
        ratio = np.median(innov_vars[valid] / expected_vars[valid])
        inflation = np.sqrt(np.clip(ratio, 0.7, 1.5))
    else:
        inflation = 1.0
    
    # Apply inflation around the analysis mean
    new_mean = np.mean(result, axis=0)
    result = new_mean + inflation * (result - new_mean)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

# EVOLVE-BLOCK-END
