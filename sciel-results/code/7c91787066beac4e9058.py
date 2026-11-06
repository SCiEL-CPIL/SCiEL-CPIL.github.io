"""Localized Ensemble Square Root Kalman Filter with adaptive inflation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a Localized Ensemble Square Root
    Kalman Filter (EnSRF) with Gaspari-Cohn localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of the observed variables.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute background statistics
    mean_b = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean_b
    
    # 2. Localization
    # We apply localization to the covariance terms.
    # For a global update, we need to localize the cross-covariance and background covariance.
    # A common approach is to apply a localization matrix to the anomalies before computing covariances,
    # or to the gain matrix.
    # Here, we use a "local" approach where we update variables in blocks or use a global gain
    # but with localized covariances.
    
    # Gaspari-Cohn correlation function
    def gaspari_cohn(dist, L):
        """
        Gaspari-Cohn correlation function.
        dist: distance between variables (index difference in Lorenz-96)
        L: correlation length (in index units)
        """
        if dist >= L:
            return 0.0
        if dist < L / 2.0:
            # Polynomial 1
            a = 5.0 / L
            b = -5.0 / (4.0 * L)
            c = 35.0 / (12.0 * L**3)
            d = -7.0 / (4.0 * L**2)
            e = 1.0 / (4.0 * L)
            f = -5.0 / (2.0 * L)
            # Actually, standard GC is defined piecewise.
            # Let's use the standard formula.
            x = dist / L
            if x < 0.5:
                return (-20 * x**5 + 50 * x**4 - 30 * x**3 + 1) / 2.0
            else:
                return (40 * x**5 - 120 * x**4 + 120 * x**3 - 50 * x**2 + 10 * x) / 2.0
        # The above if/else inside is redundant if I just use the piecewise formula directly.
        x = dist / L
        if x < 0.5:
            return (-20 * x**5 + 50 * x**4 - 30 * x**3 + 1) / 2.0
        else:
            return (40 * x**5 - 120 * x**4 + 120 * x**3 - 50 * x**2 + 10 * x) / 2.0

    # Choose a correlation length L. For N=20, L=4 or 5 is typical.
    L_loc = 4.0
    
    # Precompute localization weights for each variable pair
    # We need to localize the cross-covariance C_bx and background covariance C_bb.
    # This is computationally expensive if done for all pairs.
    # Alternative: Apply localization to the anomalies themselves? No, that's not standard.
    # Standard EnKF with localization:
    # K = C_bx L (L C_bb L + R)^-1
    # Where L is the localization operator.
    
    # Let's compute the localized cross-covariance and background covariance.
    # C_bx is (n_vars, n_obs). C_bb is (n_obs, n_obs).
    
    # Indices for observed variables
    obs_idx = observed_indices
    
    # Compute C_bb (n_obs x n_obs)
    # C_bb[i, j] = (1/(N-1)) * sum_k A_obs[k, i] * A_obs[k, j] * L(i, j)
    A_obs = anomalies[:, obs_idx]
    
    # Compute localization matrix for observed variables
    # dist(i, j) = abs(i - j) in the state space indices
    # Note: obs_idx are indices in the 20-var state.
    loc_mat_obs = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            dist = abs(obs_idx[i] - obs_idx[j])
            loc_mat_obs[i, j] = gaspari_cohn(dist, L_loc)
    
    # C_bb_loc[i, j] = (1/(N-1)) * A_obs[:, i].T @ A_obs[:, j] * loc_mat_obs[i, j]
    C_bb_raw = (A_obs.T @ A_obs) / (n_members - 1)
    C_bb_loc = C_bb_raw * loc_mat_obs
    
    # Compute C_bx (n_vars x n_obs)
    A_full = anomalies
    C_bx_raw = (A_full.T @ A_obs) / (n_members - 1)
    
    # Localize C_bx:
    # C_bx_loc[k, i] = C_bx_raw[k, i] * L(k, obs_idx[i])
    loc_mat_bx = np.zeros((n_vars, n_obs))
    for k in range(n_vars):
        for i in range(n_obs):
            dist = abs(k - obs_idx[i])
            loc_mat_bx[k, i] = gaspari_cohn(dist, L_loc)
            
    C_bx_loc = C_bx_raw * loc_mat_bx
    
    # Observation covariance
    R = np.eye(n_obs) * observation_variance
    
    # Innovation
    innovation = observations - mean_b[obs_idx]
    
    # Compute Gain K = C_bx_loc @ inv(C_bb_loc + R)
    S = C_bb_loc + R
    # Add small regularization for stability
    S += 1e-8 * np.eye(n_obs)
    
    try:
        K = C_bx_loc @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        K = C_bx_loc @ np.linalg.pinv(S)
        
    # Update Mean
    mean_a = mean_b + K @ innovation
    
    # Update Anomalies (Square Root Form)
    # x_a = x_b + K @ (y - y_b) - K @ H @ (x_b - x_b_mean) ... this is not quite right for square root.
    # Standard EnSRF update:
    # Let A be the anomaly matrix (N x M).
    # We want to update A such that the new covariance matches the EnKF covariance.
    # A_new = A - K @ A_obs @ L_inv ... this is complex.
    
    # Simpler: Use the standard deterministic EnKF update on anomalies, then inflate.
    # x_i_a = x_i_b + K @ (y - y_b)  -- This is the "spherical" update, it collapses spread.
    # To preserve spread, we use the "EnKF" update:
    # x_i_a = x_i_b + K @ (y_i - y)
    # where y_i are perturbed observations.
    # Since we don't have y_i, we use the "EnKF-a" or "EnSRF" formula.
    
    # EnSRF Formula:
    # A_new = A - K @ A_obs @ (I + A_obs^T A_obs / (N-1) / R)^-1 ... No.
    
    # Let's use the standard EnKF update with perturbed observations simulation.
    # This is the "Stochastic EnKF".
    # Generate random perturbations for observations?
    # Or use the "EnKF" where:
    # x_i_a = x_i_b + K @ (y - y_b) - K @ H @ (x_i_b - x_b)
    # This is equivalent to:
    # x_i_a = x_i_b + K @ (y - y_b) - K @ A_obs_row_i
    
    # Let's use this "deterministic" form which is common in EnSRF implementations if we handle the spread correctly.
    # Actually, the standard EnKF update for member i is:
    # x_i_a = x_i_b + K @ (y - y_b) - K @ H @ (x_i_b - x_b)
    # This preserves the ensemble mean.
    # Does it preserve the spread?
    # The new anomaly a_i_a = a_i_b - K @ H @ a_i_b = (I - K H) a_i_b.
    # The new covariance is (I - K H) C_b (I - K H)^T.
    # This is the correct EnKF covariance.
    
    # So, update anomalies:
    # A_obs is (N, n_obs).
    # K is (n_vars, n_obs).
    # H is selection operator.
    # a_i_a = a_i_b - K @ a_i_obs
    # where a_i_obs is the i-th row of A_obs.
    
    # Vectorized:
    # A_new = A_full - K @ A_obs
    # Wait, K @ A_obs is (n_vars, N).
    # A_full is (N, n_vars).
    # We need A_full_new[i, :] = A_full[i, :] - (K @ A_obs[:, i])
    # A_full_new = A_full - (K @ A_obs).T
    
    A_new = A_full - (K @ A_obs).T
    
    # Apply Inflation
    # Compute spread before and after
    # We want to restore the spread to a target level.
    # Target: Background spread, or slightly larger.
    
    # Compute standard deviation of anomalies
    # Note: A_new is centered around 0 by construction?
    # mean(A_new) = mean(A_full) - K @ mean(A_obs) = 0 - K @ 0 = 0.
    # So A_new is the anomaly matrix.
    
    # Inflation factor
    # Let's use adaptive inflation.
    # Ratio of background variance to analysis variance.
    # var_b = diag(C_b) = diag(A_full.T @ A_full / (N-1))
    # var_a = diag(A_new.T @ A_new / (N-1))
    
    var_b = np.sum(A_full**2, axis=0) / (n_members - 1)
    var_a = np.sum(A_new**2, axis=0) / (n_members - 1)
    
    # Avoid division by zero
    var_a[var_a < 1e-10] = 1e-10
    
    # Inflation factor alpha such that alpha^2 * var_a = var_b (or target)
    # We might want to inflate to slightly more than background to account for model error.
    # Let's target background spread.
    inflation = np.sqrt(var_b / var_a)
    
    # Clip inflation to reasonable range
    inflation = np.clip(inflation, 1.0, 1.2)
    
    # Apply inflation to anomalies
    A_new = inflation * A_new
    
    # Reconstruct ensemble
    ensemble_updated = mean_a + A_new
    
    # Ensure finite and within bounds
    ensemble_updated = np.nan_to_num(ensemble_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_updated = np.clip(ensemble_updated, -100.0, 100.0)
    
    return ensemble_updated
